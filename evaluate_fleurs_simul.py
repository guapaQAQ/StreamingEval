import os
import argparse
import logging
import torch
import json
import csv
from pathlib import Path
from tqdm import tqdm
from seamless_communication.datasets.huggingface import Speech2SpeechFleursDatasetBuilder
from simul_whisper.simul_whisper.transcriber.config import AlignAttConfig
from simul_whisper.simul_whisper.transcriber.segment_loader import SegmentWrapper
from simul_whisper.simul_whisper.transcriber.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
from simul_whisper.simul_whisper.transcriber.latency_scorer import (
    ALScorer, LAALScorer, APScorer, DALScorer, ATDScorer, Instance
)
from mapping import UNITY_TO_FLEURS_LANG_MAPPING, UNITY_TO_WHISPER_LANG_MAPPING
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("simul_evaluation")

class CustomInstance:
    """Custom instance class for tracking transcription progress and metrics.
    Follows the exact format from instances.log.
    """
    
    def __init__(self, interval: float, reference: str = None, source: str = None, sample_rate: int = 16000):
        """Initialize instance with interval size and reference text.
        
        Args:
            interval: Time interval between segments in seconds
            reference: Reference text for evaluation
            source: Source audio path
            sample_rate: Audio sample rate (default: 16000 for Whisper)
        """
        self.interval = interval
        self.reference = reference
        self.latency_unit = "spm"  # Set the latency unit to SPM
        self.target_spm_model = None  # Initialize the target SPM model
        self.source_length = 0  # Will be set based on audio length
        self.delays = []  # List of delays for each segment (unaware latency in ms)
        self.elapsed = []  # List of elapsed times for each prediction (aware latency in ms)
        self.token_chunk_id = []  # List of token chunk IDs
        self.prediction_list = []  # List of predicted tokens
        self.metrics = {}  # Dictionary of metrics
        self.start_time = time.time()  # Initialize start time immediately
        self.finish_prediction = False
        self.index = None  # Will be set when saving
        self.source = source  # Source audio path
        self.sample_rate = sample_rate  # Audio sample rate
        self.final_hypothesis = None  # Store the final decoded hypothesis text
        self.last_elapsed = 0.0
        self.reference_spm = None  # Store reference SPM tokens
        
    def step_to_elapsed(self, segment_id: int, current_time: float) -> float:
        """Calculate elapsed time in milliseconds (aware latency).
        
        Args:
            step: Current step number
            current_time: Current timestamp
            
        Returns:
            Elapsed time in milliseconds
        """
        # For aware latency, we only consider computation time
        return max(self.len_sample_to_ms(segment_id), self.last_elapsed) + (current_time - self.start_time) * 1000
        
    def len_sample_to_ms(self, segment_id: int) -> float:
        """Convert segment ID to milliseconds (unaware latency).
        
        Args:
            segment_id: ID of the current segment
            
        Returns:
            Delay in milliseconds
        """
        return (segment_id+1)*self.interval*1000
        
    def append_segment(
        self,
        tokens: List[int],
        segment_id: int,
    ) -> None:
        """Append a new segment of tokens.
        
        Args:
            tokens: List of token IDs for this segment
            segment_id: ID of the current segment
        """
        # Calculate delay based on segment position (unaware latency)
        delay = self.len_sample_to_ms(segment_id)
        
        # Calculate elapsed time in milliseconds (aware latency)
        current_time = time.time()
        elapsed = self.step_to_elapsed(segment_id, current_time)
        self.last_elapsed = elapsed
        
        # Add timing information - repeat elapsed time for each token
        self.delays.extend([delay] * len(tokens))
        self.elapsed.extend([elapsed] * len(tokens))
        
        # Add tokens
        self.token_chunk_id.append(segment_id)
        self.prediction_list.extend(tokens)
        
        
    @property
    def prediction(self) -> str:
        """Get the current prediction as a string."""
        return " ".join(map(str, self.prediction_list))  # Ensure this returns the final hypothesis
    
    @property
    def prediction_length(self) -> int:
        """Get the length of the current prediction."""
        return len(self.prediction_list)
    
    def summarize(self) -> Dict:
        """Create a summary of the instance for logging.
        Follows the exact format from instances.log.
        
        Returns:
            Dictionary containing instance information
        """
        return {
            "index": self.index,
            "prediction": self.final_hypothesis if self.final_hypothesis is not None else self.prediction,
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.prediction_length,
            "reference": self.reference,
            "reference_spm": self.reference_spm,  # Use stored SPM tokens
            "source": [f"samplerate: {self.sample_rate}.0", f"path: {self.source}"],
            "source_length": self.source_length,
            "prediction_spm": self.prediction_list  # Add SPM tokens
        }

    def set_target_spm_model(self, spm_model):
        """Set the target SPM model and encode reference text.
        This should be called after all delays are counted to ensure fair latency measurements.
        """
        self.target_spm_model = spm_model
        # Encode reference text after delays are counted
        if self.reference is not None:
            self.reference_spm = self.target_spm_model.encode(self.reference)

    @property
    def reference_length(self) -> int:
        if self.latency_unit == "spm":
            assert self.reference_spm is not None, "Reference SPM tokens not set"
            return len(self.reference_spm)
        else:
            return len(self.reference.split())  # Default to word count if not SPM

def transcribe_audio(
    model: PaddedAlignAttWhisper,
    audio_path: str,
    segment_length: float,
    reference: str
) -> Tuple[str, CustomInstance]:
    """Transcribe audio using simul_whisper.
    
    Args:
        model: The simul_whisper model instance
        audio_path: Path to the audio file
        segment_length: Length of each segment in seconds
        reference: Reference text for evaluation
        
    Returns:
        Tuple containing:
        - hypothesis: The transcribed text
        - instance: CustomInstance object containing latency information
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If transcription fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        segmented_audio = SegmentWrapper(audio_path=audio_path, segment_length=segment_length)
        # Get audio duration in ms
        audio_len_s = segmented_audio.audio_len_s
        duration_ms = audio_len_s * 1000
        sample_rate = 16000
        
        hyp_list = []
        instance = CustomInstance(
            interval=segment_length,
            reference=reference,
            source=audio_path,
            sample_rate=sample_rate  # Whisper uses 16kHz audio
        )
        instance.source_length = duration_ms
        
        for seg_id, (seg, is_last) in enumerate(segmented_audio):
            try:
                new_toks = model.infer(seg, is_last)
                hyp_list.append(new_toks)
                hyp = torch.cat(hyp_list, dim=0)
                hyp = hyp[hyp < DEC_PAD]
                hyp = model.tokenizer.decode(hyp)
                
                # Record delays for latency measurement
                instance.append_segment(new_toks.tolist(), seg_id)
                instance.start_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing segment {seg_id}: {str(e)}")
                raise RuntimeError(f"Failed to process segment {seg_id}: {str(e)}")
        
        model.refresh_segment(complete=True)
        instance.finish_prediction = True
        instance.final_hypothesis = hyp  # Store the final decoded hypothesis text
        
        # Set SPM model and encode reference after all delays are counted
        instance.set_target_spm_model(model.tokenizer)
        
        return hyp, instance
        
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def save_results(results: List[Dict[str, str]], output_file: str) -> None:
    """Save results to a JSON file.
    
    Args:
        results: List of dictionaries containing results
        output_file: Path to output file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {str(e)}")
        raise IOError(f"Failed to save results: {str(e)}")

def save_metrics(metrics: Dict[int, Dict[str, float]], output_file: str) -> None:
    """Save metrics to a TSV file.
    
    Args:
        metrics: Dictionary mapping sample IDs to their metrics
        output_file: Path to output file
        
    Raises:
        IOError: If file cannot be written
    """
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

def save_instances(instances: Dict[int, CustomInstance], output_file: str) -> None:
    """Save detailed instance information to a log file.
    
    Args:
        instances: Dictionary mapping sample IDs to their instances
        output_file: Path to output file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample_id, instance in sorted(instances.items()):
                # Set the index before summarizing
                instance.index = sample_id
                # Write as JSON with proper formatting
                json.dump(instance.summarize(), f, indent=2, ensure_ascii=False)
                f.write("\n")  # Add newline between instances
        logger.info(f"Instances saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save instances to {output_file}: {str(e)}")
        raise IOError(f"Failed to save instances: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate simul_whisper performance")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory to store dataset and audio files")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save evaluation results")
    parser.add_argument("--source-lang", type=str, required=True,
                      help="Source language code (e.g., 'eng' for English)")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to the whisper checkpoint")
    parser.add_argument("--if-ckpt-path", type=str, required=True,
                      help="Path to the CIF model checkpoint")
    parser.add_argument("--segment-length", type=float, default=1.0,
                      help="Chunk length in seconds")
    parser.add_argument("--frame-threshold", type=int, default=12,
                      help="Threshold for attention-guided decoding in frames")
    parser.add_argument("--buffer-len", type=int, default=20,
                      help="Length of context buffer in seconds")
    parser.add_argument("--min-seg-len", type=float, default=0.0,
                      help="Minimum segment length threshold")
    parser.add_argument("--language", type=str, default="en",
                      help="Language code")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help="Set the log level")
    
    args = parser.parse_args()
    
    # Set up logging
    logger.setLevel(args.log_level)
    
    try:
        # Check language code mappings
        if args.source_lang not in UNITY_TO_FLEURS_LANG_MAPPING:
            raise ValueError(
                f"No FLEURS language code mapping for {args.source_lang}. "
                f"Available languages: {', '.join(sorted(UNITY_TO_FLEURS_LANG_MAPPING.keys()))}"
            )
        
        if args.source_lang not in UNITY_TO_WHISPER_LANG_MAPPING:
            raise ValueError(
                f"No Whisper language code mapping for {args.source_lang}. "
                f"Available languages: {', '.join(sorted(UNITY_TO_WHISPER_LANG_MAPPING.keys()))}"
            )
        
        fleurs_lang = UNITY_TO_FLEURS_LANG_MAPPING[args.source_lang]
        whisper_lang = UNITY_TO_WHISPER_LANG_MAPPING[args.source_lang]
        logger.info(f"Using FLEURS language code: {fleurs_lang} for {args.source_lang}")
        logger.info(f"Using Whisper language code: {whisper_lang} for {args.source_lang}")
        
        # Create language-specific output directory
        lang_output_dir = os.path.join(args.output_dir, f"{args.source_lang}_fleurs_test")
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model...")
        cfg = AlignAttConfig(
            model_path=args.model_path,
            segment_length=args.segment_length,
            frame_threshold=args.frame_threshold,
            language=whisper_lang,  # Use whisper_lang instead of args.language
            buffer_len=args.buffer_len,
            min_seg_len=args.min_seg_len,
            if_ckpt_path=args.if_ckpt_path,
        )
        model = PaddedAlignAttWhisper(cfg)
        logger.info("Model initialized successfully")
        
        # Initialize latency scorers
        latency_scorers = {
            'AL': ALScorer(),
            'LAAL': LAALScorer(),
            'AP': APScorer(),
            'DAL': DALScorer(),
            'ATD': ATDScorer()
        }
        
        # Initialize dataset builder
        logger.info("Initializing dataset...")
        try:
            dataset_iterator = Speech2SpeechFleursDatasetBuilder(
                source_lang=fleurs_lang,
                target_lang=fleurs_lang,
                split="test",
                dataset_cache_dir=os.path.join(args.data_dir, args.source_lang),
                skip_source_audio=False,
                skip_target_audio=True,
            )
            logger.info("Dataset initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dataset builder for {args.source_lang} ({fleurs_lang}): {str(e)}")
            raise
        
        # Set up output files
        output_file = os.path.join(lang_output_dir, "results.tsv")
        metrics_file = os.path.join(lang_output_dir, "metrics.tsv")
        instances_file = os.path.join(lang_output_dir, "instances.log")
        
        # Load existing results if any
        results = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    results = list(reader)
                logger.info(f"Loaded {len(results)} existing results from {output_file}")
            except Exception as e:
                logger.error(f"Error loading existing results: {str(e)}")
                results = []
        
        # Get the last processed index
        last_processed_idx = max([int(r['id']) for r in results]) if results else -1
        
        # Process each audio file
        metrics = {}
        instances = {}
        
        for idx, sample in enumerate(tqdm(dataset_iterator, desc="Processing audio files")):
            # Skip already processed samples
            if idx <= last_processed_idx:
                continue
                
            audio_path = sample.source.audio_local_path
            reference_text = sample.source.text
            
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
                
            try:
                logger.info(f"Processing sample {idx}")
                hypothesis, instance = transcribe_audio(model, audio_path, args.segment_length, reference_text)
                
                # Calculate latency metrics
                latency_metrics = {}
                for metric_name, scorer in latency_scorers.items():
                    try:
                        score = scorer({0: instance})
                        latency_metrics[metric_name] = score
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
                        latency_metrics[metric_name] = None
                
                # Store results
                results.append({
                    'id': str(idx),
                    'reference': reference_text,
                    'hypothesis': hypothesis
                })
                
                # Store metrics and instances
                metrics[idx] = latency_metrics
                instances[idx] = instance
                
                # Save results after each sample
                try:
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['id', 'reference', 'hypothesis'], delimiter='\t')
                        writer.writeheader()
                        writer.writerows(results)
                    
                    # Save metrics and instances
                    save_metrics(metrics, metrics_file)
                    save_instances(instances, instances_file)
                    
                    logger.info(f"Successfully processed and saved sample {idx}")
                except Exception as e:
                    logger.error(f"Failed to save results for sample {idx}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Calculate average latency metrics across all samples
        avg_latency_metrics = {}
        for metric_name in latency_scorers.keys():
            valid_scores = [m[metric_name] for m in metrics.values() 
                          if m[metric_name] is not None]
            if valid_scores:
                avg_latency_metrics[metric_name] = sum(valid_scores) / len(valid_scores)
            else:
                avg_latency_metrics[metric_name] = None
        
        # Save summary with average metrics
        summary_file = os.path.join(lang_output_dir, "scores.tsv")
        try:
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['metric', 'value'])
                writer.writerow(['total_samples', len(results)])
                for metric, value in avg_latency_metrics.items():
                    writer.writerow([metric, value])
            
            logger.info(f"Evaluation completed for {args.source_lang}. Total samples processed: {len(results)}")
            logger.info(f"Average latency metrics: {avg_latency_metrics}")
        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 