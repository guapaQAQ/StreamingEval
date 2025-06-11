# StreamingEval

A comprehensive evaluation suite for streaming Automatic Speech Recognition (ASR) systems on the FLEURS dataset. This repository provides tools to evaluate and compare different streaming ASR models including Seamless, Whisper, and SimulEval across multiple languages and configurations.

## Overview

StreamingEval evaluates streaming ASR systems on their accuracy and latency performance using the FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) dataset. The suite supports:

- **Seamless Communication**: Meta's streaming multilingual ASR system
- **Whisper**: OpenAI's speech recognition model with streaming capabilities  
- **SimulEval**: Simultaneous evaluation framework with Whisper backend

## Features

- 🌍 **Multi-language Support**: Evaluate on 30+ languages across 7 language families
- 📊 **Comprehensive Metrics**: Quality metrics (BLEU, ASR-BLEU) and latency metrics (AL, LAAL, AP, DAL, ATD)
- 🔄 **Streaming Evaluation**: Real-time performance assessment with configurable chunk sizes
- 📝 **Detailed Logging**: Individual logs per language/configuration plus summary reports
- 🎯 **Flexible Configuration**: Run specific evaluations or full suite with customizable parameters
- 💾 **Resumable Execution**: Skip already completed evaluations automatically

## Quick Start

### Prerequisites

First, clone the required repositories:

```bash
# Clone SimulEval
git clone https://github.com/facebookresearch/SimulEval.git

# Clone Seamless Communication
git clone https://github.com/facebookresearch/seamless_communication.git

# Clone Simul-Whisper
git clone https://github.com/backspacetg/simul_whisper.git
```

Set up the environment:

```bash
# Create and activate conda environment
conda create -n seamless python=3.8
conda activate seamless

# Install PyTorch and related packages
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# Install fairseq (specific version)
pip install fairseq==0.12.2

# Install additional requirements
pip install datasets
pip install accelerate
pip install transformers
pip install librosa
pip install soundfile
pip install openai-whisper

# Install SimulEval
cd SimulEval
pip install -e .
cd ..

# Install Seamless Communication
cd seamless_communication
pip install -e .
cd ..

# Install Simul-Whisper dependencies
cd simul_whisper
pip install -r requirements.txt
cd ..
```

### Basic Usage

```bash
# Run all evaluations on core languages (default)
./run_evaluation_suite.sh

# Run specific evaluation type
./run_evaluation_suite.sh --type seamless
./run_evaluation_suite.sh --type whisper  
./run_evaluation_suite.sh --type simul

# Use representative language set (30+ languages)
./run_evaluation_suite.sh --languages representative

# Custom directories
./run_evaluation_suite.sh --data-dir /path/to/data --output-dir /path/to/results
```

## Evaluation Systems

### 1. Seamless Communication

Evaluates Meta's Seamless streaming ASR system with:
- Multilingual speech recognition
- Real-time processing capabilities
- Built-in quality and latency metrics

**Configuration:**
- Model: `seamless_streaming_unity`
- Segment size: 320ms
- Early stopping: Disabled
- Device: CUDA with FP16 precision

### 2. Whisper Streaming

Evaluates OpenAI's Whisper with streaming modifications:
- Configurable chunk sizes
- VAD (Voice Activity Detection) support
- Buffer trimming strategies

**Configuration:**
- Model: `large-v2`
- Backend: `faster-whisper`
- Chunk size: 1.0s (configurable)
- VAD: Enabled
- Buffer trimming: 15.0s

### 3. SimulEval Framework

Evaluates simultaneous translation/transcription using:
- Whisper backend with CIF (Continuous Integrate-and-Fire) models
- Attention-guided decoding
- Configurable latency thresholds

**Configuration:**
- Model: `large-v2`
- CIF model: `simul_whisper/cif_models/large-v2.pt`
- Segment length: 0.5s (configurable)
- Frame threshold: 12
- Buffer length: 20s

## Language Support

### Core Languages (7 languages)
Representative languages from different families:
- **eng** (English) - Western Europe
- **rus** (Russian) - Eastern Europe  
- **tur** (Turkish) - Central Asia
- **swa** (Swahili) - Sub-Saharan Africa
- **hin** (Hindi) - South Asia
- **tha** (Thai) - South East Asia
- **jpn** (Japanese) - CJK

### Representative Languages (30+ languages)
Full coverage across language families:
- **Western Europe**: eng, spa, deu, fra, ita, nld
- **Eastern Europe**: rus, pol, ces, ukr
- **Central Asia**: ara, tur, fas, heb
- **Sub-Saharan Africa**: swa, hau, yor, zul, amh
- **South Asia**: hin, ben, tam, pan
- **South East Asia**: ind, tha, vie, fil
- **CJK**: zho, jpn, kor

## Command Line Options

```bash
Usage: ./run_evaluation_suite.sh [OPTIONS]

Options:
  --type TYPE          Evaluation type: seamless, whisper, simul, or all (default: all)
  --languages SET      Language set: core or representative (default: core)
  --data-dir DIR       Data directory (default: data/fleurs)
  --output-dir DIR     Base output directory (default: evaluation_outputs)
  --help               Show help message
```

## Output Structure

```
evaluation_outputs/
├── seamless/
│   ├── eng_fleurs_test/
│   │   ├── results.tsv
│   │   ├── metrics.tsv
│   │   └── scores.tsv
│   └── ...
├── whisper/
│   ├── eng_chunk1.0_fleurs_test/
│   │   ├── results.tsv
│   │   ├── metrics.tsv
│   │   └── instances.log
│   └── ...
└── simul/
    ├── eng_seg0.5_fleurs_test/
    │   ├── results.tsv
    │   ├── metrics.tsv
    │   └── instances.log
    └── ...

evaluation_logs/
├── evaluation_suite_summary.txt
├── eng_seamless.log
├── eng_chunk1.0_whisper.log
├── eng_seg0.5_simul.log
└── ...
```

## Metrics

### Quality Metrics
- **ASR-BLEU**: ASR-specific BLEU score
- **Hypothesis**: Generated transcription
- **Reference**: Ground truth transcription

### Latency Metrics
- **AL (Average Lagging)**: Average delay between input and output
- **LAAL (Length Adaptive Average Lagging)**: Length-normalized average lagging
- **AP (Average Proportion)**: Proportion of source processed when generating output
- **DAL (Differentiable Average Lagging)**: Differentiable version of AL
- **ATD (Average Token Delay)**: Average delay per token

## Individual Evaluation Scripts

### evaluate_fleurs_seamless.py
```bash
python evaluate_fleurs_seamless.py \
    --source-lang eng \
    --data-dir data/fleurs \
    --output-dir results/seamless/eng_test
```

### evaluate_fleurs_whisper.py
```bash
python evaluate_fleurs_whisper.py \
    --source-lang eng \
    --data-dir data/fleurs \
    --output-dir results/whisper/eng_test \
    --model large-v2 \
    --min-chunk-size 1.0
```

### evaluate_fleurs_simul.py
```bash
python evaluate_fleurs_simul.py \
    --source-lang eng \
    --data-dir data/fleurs \
    --output-dir results/simul/eng_test \
    --model-path large-v2 \
    --if-ckpt-path simul_whisper/cif_models/large-v2.pt \
    --segment-length 0.5
```

## Advanced Configuration

### Custom Model Configurations

Edit the configuration variables in `run_evaluation_suite.sh`:

```bash
# Whisper Configuration
WHISPER_MODEL="large-v3"
WHISPER_CHUNK_SIZES=(0.5 1.0 2.0)

# SimulEval Configuration  
SIMUL_SEGMENT_LENGTHS=(0.25 0.5 1.0)
SIMUL_FRAME_THRESHOLD=8

# Test specific languages
CUSTOM_LANGUAGES=("eng" "spa" "fra")
```

### Parallel Processing

For faster evaluation, you can run different evaluation types in parallel:

```bash
# Terminal 1
./run_evaluation_suite.sh --type seamless --languages representative &

# Terminal 2  
./run_evaluation_suite.sh --type whisper --languages representative &

# Terminal 3
./run_evaluation_suite.sh --type simul --languages representative &
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Missing Model Files**: Ensure all model checkpoints are downloaded
3. **Dataset Download Failures**: Check internet connection and disk space
4. **Permission Errors**: Ensure script has execute permissions (`chmod +x run_evaluation_suite.sh`)

### Debug Mode

Add debug logging by modifying the Python scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-evaluation`)
3. Commit your changes (`git commit -am 'Add new evaluation system'`)
4. Push to the branch (`git push origin feature/new-evaluation`)
5. Create a Pull Request

## Citation

If you use this evaluation suite in your research, please cite the relevant papers:

### SimulEval
```bibtex
@inproceedings{ma-etal-2020-simuleval,
    title = "{S}imul{E}val: An Evaluation Toolkit for Simultaneous Translation",
    author = "Ma, Xutai  and
      Mohammad, Fahim  and
      Kocabiyikoglu, Can  and
      Lee, Ann  and
      Tran, Ke  and
      Watanabe, Shinji  and
      Pino, Juan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.12",
    doi = "10.18653/v1/2020.emnlp-demos.12",
    pages = "77--84"
}
```

### Whisper
```bibtex
@article{radford2022whisper,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

### Streaming Whisper
```bibtex
@inproceedings{whisper_streaming,
  title={Whisper streaming},
  author={Polák, Dominik and Kocour, Martin and Burget, Lukáš and Černocký, Jan},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},  
  year={2023},
  organization={IEEE}
}
```

### Simul-Whisper
```bibtex
@inproceedings{guo2023simulwhisper,
  title={Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection},
  author={Guo, Haoyu and Chen, Shujie and Ma, Minghan and Zhang, Jiaxin and Wong, Derek F and Chao, Lidia S},
  booktitle={Proceedings of INTERSPEECH 2023},
  year={2023}
}
```

### SeamlessM4T
```bibtex
@article{barrault2023seamlessm4t,
  title={SeamlessM4T—Massively Multilingual \& Multimodal Machine Translation},
  author={Barrault, Loïc and Chung, Yu-An and Meglioli, Mariano Coria and David, Elahe and Dabre, Raj and Duh, Kevin and Duquenne, Paul-Ambroise and Durmus, Necip Fazil and Edunov, Sergey and others},
  journal={arXiv preprint arXiv:2308.11596},
  year={2023}
}
```

### FLEURS Dataset
```bibtex
@article{conneau2022fleurs,
  title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FLEURS dataset creators
- Seamless Communication team at Meta
- OpenAI Whisper team
- SimulEval framework developers