import os
import argparse
import logging
from pathlib import Path
import torch
import time
import math
from seamless_communication.datasets.huggingface import Speech2SpeechFleursDatasetBuilder
# from seamless_communication.cli.m4t.finetune.dataset import UNITY_TO_FLEURS_LANG_MAPPING
from mapping import UNITY_TO_FLEURS_LANG_MAPPING
from simuleval.cli import evaluate
from simuleval.evaluator import SentenceLevelEvaluator
from simuleval.evaluator.instance import SpeechToTextInstance, SpeechSegment, EmptySegment, INSTANCE_TYPE_DICT
from simuleval.evaluator.scorers import get_scorer_class
from simuleval.evaluator.scorers.latency_scorer import (
    ALScorer, LAALScorer, APScorer, DALScorer, ATDScorer
)
from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent
from seamless_communication.cli.streaming.scorers.seamless_quality_scorer import SeamlessQualityScorer
from fairseq2.assets import asset_store, download_manager
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("asr_evaluation")

class CustomASRInstance(SpeechToTextInstance):
    """Custom instance class for tracking ASR processing information."""
    
    def __init__(self, index: int, dataloader=None, args=None):
        super().__init__(index, dataloader, args)
        self.last_elapsed = 0.0
        self.start_time = time.time()
        # logger.info(f"Initializing CustomASRInstance for index {index}")
        
    def step_to_elapsed(self, step: int, current_time: float) -> float:
        """Calculate elapsed time in milliseconds (aware latency).
        
        Args:
            step: Current step number
            current_time: Current timestamp
            
        Returns:
            Elapsed time in milliseconds
        """
        # For aware latency, we only consider computation time
        return max(self.len_sample_to_ms(step), self.last_elapsed) + (current_time - self.start_time) * 1000
            
    
    def send_source(self, segment_size=10):
        # if self.step == 0: # Should always set the timer
        self.start_time = time.time()
        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                # Pad zeros if the requested number of samples
                # are more than available samples.
                samples = self.samples[self.step :]  # noqa E203
                is_finished = True
                self.source_finished_reading = True
            else:
                samples = self.samples[self.step : self.step + num_samples]  # noqa E203
                is_finished = False

            self.step = min(self.step + num_samples, len(self.samples))

            segment = SpeechSegment(
                index=self.len_sample_to_ms(self.step),
                content=samples,
                sample_rate=self.audio_info.samplerate,
                finished=is_finished,
                tgt_lang=self.tgt_lang,
            )

        else:
            # Finish reading this audio
            segment = EmptySegment(
                index=self.len_sample_to_ms(self.step),
                finished=True,
            )
            self.source_finished_reading = True

        return segment

# Register our custom instance class with SimulEval
INSTANCE_TYPE_DICT["speech-text"] = CustomASRInstance

def prepare_asr_data(
    source_lang: str,
    split: str,
    data_dir: str,
):
    """Prepare ASR data from FLEURS dataset and save in TSV format."""
    # Check language code mapping
    if source_lang not in UNITY_TO_FLEURS_LANG_MAPPING:
        raise ValueError(
            f"No language code mapping for {source_lang}. "
            "Please check UNITY_TO_FLEURS_LANG_MAPPING"
        )
    
    # Create language-specific directory
    lang_dir = os.path.join(data_dir, source_lang)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Initialize dataset builder
    try:
        dataset_iterator = Speech2SpeechFleursDatasetBuilder(
            source_lang=UNITY_TO_FLEURS_LANG_MAPPING[source_lang],
            target_lang=UNITY_TO_FLEURS_LANG_MAPPING[source_lang],  # For ASR, source and target are same
            split=split,
            dataset_cache_dir=lang_dir,
            skip_source_audio=False,  # We need source audio for ASR
            skip_target_audio=True,   # We don't need target audio for ASR
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset builder for {source_lang}: {str(e)}")
        raise
    
    # Create TSV file
    tsv_path = os.path.join(lang_dir, f"{split}_asr.tsv")
    try:
        with open(tsv_path, "w") as f:
            # Write header
            f.write("id\taudio\ttgt_text\n")
            
            # Write data
            for sample in dataset_iterator:
                # Get relative path from language directory
                rel_path = os.path.relpath(sample.source.audio_local_path, lang_dir)
                # Write TSV line with target text as a list
                f.write(f"{sample.source.id}\t{rel_path}\t{sample.source.text}\n")
        
        logger.info(f"Saved ASR data to {tsv_path}")
        return tsv_path
    except Exception as e:
        logger.error(f"Failed to write TSV file for {source_lang}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR performance on FLEURS dataset",
        add_help=False,
        conflict_handler="resolve"
    )
    parser.add_argument("--source-lang", type=str, required=True,
                      help="Source language code (e.g., 'eng' for English)")
    parser.add_argument("--split", type=str, default="test",
                      help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory to store dataset and audio files")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save evaluation results")
    
    args, _ = parser.parse_known_args()
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        
        tsv_path = prepare_asr_data(
            source_lang=args.source_lang,
            split=args.split,
            data_dir=args.data_dir,
        )
        
        model_configs = {
            "source_segment_size": 320,
            "device": "cuda:0",
            "dtype": "fp16",
            "min_starting_wait_w2vbert": 192,
            "decision_threshold": 0.5,
            "no_early_stop": True,
            "max_len_a": 0,
            "max_len_b": 100,
        }
        
        # Get tokenizer path from Unity model's asset card
        asset_card = asset_store.retrieve_card(name="seamless_streaming_unity")
        tokenizer_uri = asset_card.field("tokenizer").as_uri()
        tokenizer_path = download_manager.download_tokenizer(
            tokenizer_uri, asset_card.name, force=False, progress=True
        )
        
        eval_configs = {
            "quality_metrics": "SEAMLESS_QUALITY_SCORER",
            "latency_metrics": "AL LAAL AP DAL ATD",
            "eval_latency_unit": "spm",
            "eval_latency_spm_model": tokenizer_path
        }
        
        base_config = {
            "dataloader": "fairseq2_s2tt",
            "dataloader_class": "seamless_communication.streaming.dataloaders.s2tt.SimulEvalSpeechToTextDataloader",
            "unity_model_name": "seamless_streaming_unity",
            "task": "asr",
            "data_file": tsv_path,
            "audio_root_dir": os.path.dirname(tsv_path),
            "output": args.output_dir,
            "tgt_lang": args.source_lang,
            "ref_field": "tgt_text",
        }
        
        
        evaluate(
            SeamlessStreamingS2TAgent,
            {**base_config, **model_configs, **eval_configs},
            parser
        )
            
        logger.info(f"Successfully completed evaluation for {args.source_lang}")
        
    except Exception as e:
        logger.error(f"Error in evaluation for {args.source_lang}: {str(e)}")
        raise

if __name__ == "__main__":
    main() 