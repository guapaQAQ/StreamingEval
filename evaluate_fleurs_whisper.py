#!/usr/bin/env python3

import os
import argparse
import logging
import subprocess
from pathlib import Path
import pandas as pd
import time
import sys
import json
import csv
from seamless_communication.datasets.huggingface import Speech2SpeechFleursDatasetBuilder
from mapping import UNITY_TO_FLEURS_LANG_MAPPING, UNITY_TO_WHISPER_LANG_MAPPING
from whisper_online import (
    FasterWhisperASR, OnlineASRProcessor, load_audio, load_audio_chunk,
    asr_factory, set_logging
)
from simul_whisper.simul_whisper.transcriber.latency_scorer import (
    ALScorer, LAALScorer, APScorer, DALScorer, ATDScorer
)
from typing import Dict, List

# Set up our own logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("fleurs_whisper")

# Suppress faster_whisper logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Global start time for timing measurements
start = None

class WhisperInstance:
    """Class to track Whisper processing information for each audio file."""
    
    def __init__(self, index: int, reference: str, source_path: str, sample_rate: int = 16000):
        self.index = index
        self.reference = reference
        self.source = source_path
        self.sample_rate = sample_rate
        self.source_length = 0  # Will be set based on audio length
        self.delays = []  # List of delays for each segment (unaware latency in ms)
        self.elapsed = []  # List of elapsed times for each prediction (aware latency in ms)
        self.prediction_list = []  # List of predicted tokens
        self.start_time = None
        self.final_hypothesis = None
        self.target_spm_model = None
        self.reference_spm = None
        self.metrics = {}  # Dictionary to store latency metrics

    def set_target_spm_model(self, spm_model):
        """Set the target SPM model and encode reference text."""
        self.target_spm_model = spm_model
        if self.reference is not None:
            # Convert Encoding object to list of token IDs
            self.reference_spm = self.target_spm_model.encode(self.reference).ids

    def append_segment(self, tokens: list, unaware_now: float, aware_now: float):
        """Append a new segment of tokens with timing information."""
        # Calculate delay based on segment position (unaware latency)
        delay = unaware_now * 1000
        
        # Calculate elapsed time in milliseconds (aware latency)
        elapsed = aware_now * 1000
        
        # Add timing information
        self.delays.extend([delay] * len(tokens))
        self.elapsed.extend([elapsed] * len(tokens))
        
        # Add tokens
        self.prediction_list.extend(tokens)

    def summarize(self) -> dict:
        """Create a summary of the instance for logging."""
        return {
            "index": self.index,
            "prediction": self.final_hypothesis if self.final_hypothesis is not None else " ".join(map(str, self.prediction_list)),
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": len(self.prediction_list),
            "reference": self.reference,
            "reference_spm": self.reference_spm,
            "source": [f"samplerate: {self.sample_rate}.0", f"path: {self.source}"],
            "source_length": self.source_length,
            "prediction_spm": self.prediction_list,
            "metrics": self.metrics
        }

    @property
    def reference_length(self) -> int:
        """Get the length of the reference text in tokens."""
        if self.reference_spm is not None:
            return len(self.reference_spm)
        return len(self.reference.split())  # Fallback to word count if SPM tokens not available

def output_transcript(o, f=None, unaware_now=None, aware_now=None):
    """Output format in stdout and optionally to file is like:
    [unaware_now] [aware_now] [beg] [end] [text]
    - unaware_now: chunk end time in milliseconds
    - aware_now: emission time from beginning of processing in milliseconds
    - beg: start timestamp of the text segment in milliseconds
    - end: end timestamp of the text segment in milliseconds
    - text: segment transcript
    """
    if unaware_now is None:
        unaware_now = time.time()-start
    if o[0] is not None:
        if aware_now is not None:
            print("%1.4f %1.4f %1.0f %1.0f %s" % (unaware_now*1000, aware_now*1000, o[0]*1000,o[1]*1000,o[2]),file=sys.stderr,flush=True)
            # print("%1.4f %1.4f %1.0f %1.0f %s" % (unaware_now*1000, aware_now*1000, o[0]*1000,o[1]*1000,o[2]),file=f,flush=True)
        elif unaware_now is not None:
            now = time.time()-start
            print("%1.4f %1.4f %1.0f %1.0f %s" % (unaware_now*1000, now*1000, o[0]*1000,o[1]*1000,o[2]),file=sys.stderr,flush=True)
            # print("%1.4f %1.4f %1.0f %1.0f %s" % (unaware_now*1000, now*1000, o[0]*1000,o[1]*1000,o[2]),file=f,flush=True)
        else:
            now = time.time()-start
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=sys.stderr,flush=True)
            # print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=f,flush=True)

def save_metrics(metrics: Dict[int, Dict[str, float]], output_file: str) -> None:
    """Save metrics to a TSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            # Write header
            writer.writerow(['id', 'AL', 'LAAL', 'AP', 'DAL', 'ATD'])
            # Write metrics for each sample
            for sample_id, sample_metrics in sorted(metrics.items()):
                writer.writerow([
                    sample_id,
                    sample_metrics.get('AL', ''),
                    sample_metrics.get('LAAL', ''),
                    sample_metrics.get('AP', ''),
                    sample_metrics.get('DAL', ''),
                    sample_metrics.get('ATD', '')
                ])
        logger.info(f"Metrics saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {output_file}: {str(e)}")
        raise IOError(f"Failed to save metrics: {str(e)}")

def save_results(results: List[Dict[str, str]], output_file: str) -> None:
    """Save results to a TSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'reference', 'hypothesis'], delimiter='\t')
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {str(e)}")
        raise IOError(f"Failed to save results: {str(e)}")

def prepare_and_process_audio(
    source_lang: str,
    split: str,
    data_dir: str,
    output_dir: str,
    model: str = "large-v3",
    backend: str = "faster-whisper",
    min_chunk_size: float = 1.0,
    use_vad: bool = True,
    use_vac: bool = False,
    vac_chunk_size: float = 0.04,
    buffer_trimming: str = "segment",
    buffer_trimming_sec: float = 15.0
):
    """Prepare FLEURS dataset and process each audio file with Whisper."""
    global start
    
    # Check language code mappings
    if source_lang not in UNITY_TO_FLEURS_LANG_MAPPING:
        raise ValueError(
            f"No FLEURS language code mapping for {source_lang}. "
            f"Available languages: {', '.join(sorted(UNITY_TO_FLEURS_LANG_MAPPING.keys()))}"
        )
    
    if source_lang not in UNITY_TO_WHISPER_LANG_MAPPING:
        raise ValueError(
            f"No Whisper language code mapping for {source_lang}. "
            f"Available languages: {', '.join(sorted(UNITY_TO_WHISPER_LANG_MAPPING.keys()))}"
        )
    
    fleurs_lang = UNITY_TO_FLEURS_LANG_MAPPING[source_lang]
    whisper_lang = UNITY_TO_WHISPER_LANG_MAPPING[source_lang]
    logger.info(f"Using FLEURS language code: {fleurs_lang} for {source_lang}")
    logger.info(f"Using Whisper language code: {whisper_lang} for {source_lang}")
    
    # Create language-specific directories
    lang_dir = os.path.join(data_dir, source_lang)
    output_lang_dir = os.path.join(output_dir, f"{source_lang}_fleurs_test")
    os.makedirs(lang_dir, exist_ok=True)
    os.makedirs(output_lang_dir, exist_ok=True)
    
    # Initialize dataset builder
    try:
        dataset_iterator = Speech2SpeechFleursDatasetBuilder(
            source_lang=fleurs_lang,
            target_lang=fleurs_lang,
            split=split,
            dataset_cache_dir=lang_dir,
            skip_source_audio=False,
            skip_target_audio=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset builder for {source_lang} ({fleurs_lang}): {str(e)}")
        raise

    # Create ASR instance once using factory
    logger.info(f"Loading Whisper {model} model for {whisper_lang}...")
    # Create args object with required attributes for asr_factory
    class Args:
        def __init__(self):
            self.model = model
            self.backend = backend
            self.lan = whisper_lang
            self.vad = use_vad
            self.buffer_trimming = buffer_trimming
            self.buffer_trimming_sec = buffer_trimming_sec
            self.min_chunk_size = min_chunk_size
            self.log_level = "DEBUG"
            self.task = "transcribe"
            self.model_cache_dir = None
            self.model_dir = None
            self.vac = use_vac
            self.vac_chunk_size = vac_chunk_size

    asr_args = Args()
    asr, online = asr_factory(asr_args, logfile=sys.stderr)
    logger.info("Whisper model loaded successfully")

    # Create output files
    instances_file = os.path.join(output_lang_dir, "instances.log")
    metrics_file = os.path.join(output_lang_dir, "metrics.tsv")
    results_file = os.path.join(output_lang_dir, "results.tsv")
    
    # Initialize latency scorers
    latency_scorers = {
        'AL': ALScorer(),
        'LAAL': LAALScorer(),
        'AP': APScorer(),
        'DAL': DALScorer(),
        'ATD': ATDScorer()
    }
    
    # Initialize metrics and results
    metrics = {}
    results = []

    # Process each audio file
    for idx, sample in enumerate(dataset_iterator):
        audio_path = sample.source.audio_local_path
        reference_text = sample.source.text
        
        # Create output file path
        output_file = os.path.join(output_lang_dir, f"{idx}.txt")
        
        # Skip if already processed
        if os.path.exists(output_file):
            logger.info(f"Skipping {idx} - already processed")
            continue
        
        logger.info(f"Processing {idx}: {audio_path}")
        
        try:
            # Initialize instance for this audio file
            instance = WhisperInstance(idx, reference_text, audio_path)
            
            # with open(output_file, 'w') as f:
            # Write reference text as first line
            # f.write(f"# Reference: {reference_text}\n")
            f = None
            
            # Initialize the processor for this audio file
            online.init()
            
            # Load the audio into the LRU cache before we start the timer
            a = load_audio_chunk(audio_path, 0, 1)
            
            # Warm up the ASR because the very first transcribe takes much more time
            asr.transcribe(a)
            
            # Initialize timing variables
            beg = 0.0  # Start at 0 seconds
            start = time.time()-beg
            last_aware_now = 0.0
            
            # Process audio in computational unaware mode
            end = min_chunk_size
            duration = len(load_audio(audio_path))/16000
            instance.source_length = duration * 1000  # Convert to ms
            instance.start_time = time.time()
            
            while True:
                a = load_audio_chunk(audio_path, beg, end)
                online.insert_audio_chunk(a)
                
                try:
                    o = online.process_iter()
                except AssertionError as e:
                    pass
                else:
                    now = time.time()
                    aware_now = max(end, last_aware_now)+now-instance.start_time
                    output_transcript(o, f, unaware_now=end, aware_now=aware_now)
                    
                    # Record tokens and timing information
                    if o[2]:  # If there's text output
                        tokens = asr.model.hf_tokenizer.encode(o[2]).ids
                        instance.append_segment(tokens, end, aware_now)

                    last_aware_now = aware_now
                    instance.start_time = time.time()
                
                if end >= duration:
                    break
                
                beg = end
                if end + min_chunk_size > duration:
                    end = duration
                else:
                    end += min_chunk_size
            
            # Get final transcription
            o = online.finish()
            now = time.time()
            aware_now = max(duration, last_aware_now)+now-instance.start_time
            output_transcript(o, f, unaware_now=duration, aware_now=aware_now)
            
            # Record final tokens
            if o[2]:  # If there's text output
                tokens = asr.model.hf_tokenizer.encode(o[2]).ids
                instance.append_segment(tokens, duration, aware_now)
            
            # Get final hypothesis (concatenated tokens)
            instance.final_hypothesis = asr.model.hf_tokenizer.decode(instance.prediction_list)
            instance.set_target_spm_model(asr.model.hf_tokenizer)
            
            # Calculate latency metrics
            for metric_name, scorer in latency_scorers.items():
                try:
                    score = scorer({0: instance})
                    instance.metrics[metric_name] = score
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
                    instance.metrics[metric_name] = None
            
            # Store results
            results.append({
                'id': str(idx),
                'reference': reference_text,
                'hypothesis': instance.final_hypothesis
            })
            
            # Store metrics
            metrics[idx] = instance.metrics
            
            # Save instance information immediately after processing
            try:
                with open(instances_file, 'a', encoding='utf-8') as f:
                    json.dump(instance.summarize(), f, ensure_ascii=False)
                    f.write("\n")
                logger.info(f"Saved instance information for {idx}")
            except Exception as e:
                logger.error(f"Failed to save instance information for {idx}: {str(e)}")
            
            # Save metrics and results after each sample
            try:
                save_metrics(metrics, metrics_file)
                save_results(results, results_file)
                logger.info(f"Saved metrics and results for {idx}")
            except Exception as e:
                logger.error(f"Failed to save metrics and results for {idx}: {str(e)}")
            
            logger.info(f"Successfully processed {idx}")
            
        except Exception as e:
            logger.error(f"Failed to process {idx}: {str(e)}")
            continue

    # Calculate and save average metrics
    avg_metrics = {}
    for metric_name in latency_scorers.keys():
        valid_scores = [m[metric_name] for m in metrics.values() 
                       if m[metric_name] is not None]
        if valid_scores:
            avg_metrics[metric_name] = sum(valid_scores) / len(valid_scores)
        else:
            avg_metrics[metric_name] = None
    
    # Save summary with average metrics
    summary_file = os.path.join(output_lang_dir, "scores.tsv")
    try:
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['metric', 'value'])
            writer.writerow(['total_samples', len(results)])
            for metric, value in avg_metrics.items():
                writer.writerow([metric, value])
        
        logger.info(f"Evaluation completed for {source_lang}. Total samples processed: {len(results)}")
        logger.info(f"Average latency metrics: {avg_metrics}")
    except Exception as e:
        logger.error(f"Failed to save summary: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Process FLEURS dataset with Whisper")
    parser.add_argument("--source-lang", type=str, required=True,
                      help="Source language code (e.g., 'eng' for English)")
    parser.add_argument("--split", type=str, default="test",
                      help="Dataset split to process (default: test)")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory to store dataset and audio files")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save Whisper outputs")
    parser.add_argument("--model", type=str, default="large-v3",
                      help="Whisper model size")
    parser.add_argument("--backend", type=str, default="faster-whisper",
                      help="Whisper backend")
    parser.add_argument("--min-chunk-size", type=float, default=1.0,
                      help="Minimum chunk size in seconds")
    parser.add_argument("--no-vad", action="store_true",
                      help="Disable VAD")
    parser.add_argument("--use-vac", action="store_true",
                      help="Enable VAC")
    parser.add_argument("--vac-chunk-size", type=float, default=0.04,
                      help="VAC chunk size in seconds")
    parser.add_argument("--buffer-trimming", type=str, default="segment",
                      help="Buffer trimming strategy")
    parser.add_argument("--buffer-trimming-sec", type=float, default=15.0,
                      help="Buffer trimming threshold in seconds")
    parser.add_argument("--log-level", type=str, default="DEBUG",
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help="Set the log level")
    
    args = parser.parse_args()
    
    # Set up logging for our logger
    logger.setLevel(args.log_level)
    
    try:
        prepare_and_process_audio(
            source_lang=args.source_lang,
            split=args.split,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model=args.model,
            backend=args.backend,
            min_chunk_size=args.min_chunk_size,
            use_vad=not args.no_vad,
            use_vac=args.use_vac,
            vac_chunk_size=args.vac_chunk_size,
            buffer_trimming=args.buffer_trimming,
            buffer_trimming_sec=args.buffer_trimming_sec
        )
        logger.info(f"Successfully completed processing for {args.source_lang}")
        
    except Exception as e:
        logger.error(f"Error in processing for {args.source_lang}: {str(e)}")
        raise

if __name__ == "__main__":
    main() 