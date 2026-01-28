# 🎙️ SeediqSpeech: Low-Resource ASR for Seediq Language

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ESPnet](https://img.shields.io/badge/ESPnet-202401-green.svg)](https://github.com/espnet/espnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

[**中文說明**](README.md) | [**English**](README_en.md)

**Low-Resource Automatic Speech Recognition System for Seediq Language**

This project is based on the [ESPnet](https://github.com/espnet/espnet) framework, utilizing the self-supervised pre-trained model (XLS-R) combined with **LoRA (Low-Rank Adaptation)** for efficient fine-tuning. By integrating an **RNN Language Model (RNNLM)**, we successfully achieved high-availability end-to-end speech recognition for Seediq, an indigenous language of Taiwan, under ultra-low resource conditions.

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Model Architecture](#-model-architecture)
- [Experiment History](#-experiment-history--optimization)
- [Results](#-final-results-phase-2-final)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Citation](#-citation)

---

## 🌟 Highlights

- **Ultra-Low Resource Breakthrough**: Expanded training data from just 1 hour (Common Voice) to approx. 4 hours (via crawling), successfully reducing CER from >300% to **19.5%**.
- **Hallucination Suppression**: Solved common low-resource "looping/hallucination" issues (Insertion Error reduced to 0.7%) through Uppercase Normalization and 16kHz sampling rate correction.
- **Efficient Training**: Using LoRA to train less than 1% of parameters, making it feasible to train on a single consumer-grade GPU (RTX 4060 8GB).

---

## 🏗️ Model Architecture

| Component | Details |
|------|---------|
| **Framework** | ESPnet2 (v202401) |
| **Pretrained Model** | `facebook/wav2vec2-xls-r-300m` |
| **Frontend** | S3PRL (Self-Supervised Speech Processing) |
| **Encoder** | XLS-R Transformer (Frozen, LoRA fine-tuning only) |
| **LoRA Config** | Rank=8, Alpha=16, Target: `q_proj`, `v_proj` |
| **Decoder** | 6-layer Transformer (d_model=256, d_ff=1024) |
| **Language Model** | RNNLM (2-layer LSTM, unit=512) |
| **Tokenizer** | Character-based (42 tokens, Uppercase Only) |
| **Optimization** | AdamW (lr=5e-4, warmup=1000 steps) |

---

## 🧪 Experiment History & Optimization

The project underwent a complete iterative process from baseline testing to final optimization:

### 📅 Phase 0: Baseline & Setup ( - 1/15)
- **Attempt**: Zero-shot transfer using a Librispeech pre-trained model.
- **Result**: **Failed (WER 99.8%)**. Confirmed that acoustic features differed too significantly; fine-tuning is mandatory.
- **Fixes**: Resolved HuggingFace `transformers` version compatibility (downgraded to v4.46.3) and fixed OOM issues (adjusted `nj=4`).

### 📅 Phase 1: Initial Fine-tuning (1/17 - 1/23)
- **Data**: Common Voice only (approx. 1 hour).
- **Result**: CER 34.7%, WER 82.8%.
- **Issues**: Severe model hallucinations (High Insertion Error) and LM overfitting due to single-domain training data (Bible corpus).
- **Optimization**: Conducted Grid Search for decoding parameters (`lm_weight 0.1`, `penalty 1.0`), reducing CER to 28.8%, but still limited by data bottleneck.

### 📅 Phase 2: Data Augmentation & Refactoring (1/24 - 1/28)
- **Augmentation**: Built crawlers for "Indigenous E-Learning Center" and "NCCU Learning Materials", increasing data from 885 sentences to **3158 sentences**.
- **Critical Fixes**:
    1. **Sampling Rate**: Unified all audio to 16kHz to eliminate time-dilation issues caused by 48k/16k mixing.
    2. **Uppercase Normalization**: Discovered mixed-case sensitivity caused `<unk>` explosion. Forced all training/testing text to **uppercase**.
    3. **ID Alignment**: Fixed `RuntimeError: Keys are mismatched` by ensuring strict alignment between `wav.scp` and `text`.
    4. **LM Retraining**: Retrained RNNLM on the augmented, standardized text. PPL (Perplexity) dropped significantly from 179.27 to **8.77**.

---

## 📊 Final Results (Phase 2 Final)

Performance on the Test Set after resolving case/alignment issues and integrating the high-quality LM:

| Phase | Config | CER (%) | WER (%) | Ins (%) | Status |
|:-------:|:-------|:-------:|:-------:|:-------:|:----:|
| **Baseline** | Zero-shot | 100.0 | 99.8 | - | ❌ |
| **Phase 1** | 1hr Data (No LM) | 34.7 | 82.8 | 4.3 | ⚠️ |
| **Phase 1++**| 1hr Data + Old LM | 28.8 | 77.6 | Medium | ⚠️ |
| **Phase 2** | **3k Data + New LM** | **19.5** | **55.4** | **0.7** | ✅ |

### 🔍 Analysis
- **CER 19.5%**: Achieved the goal of high availability (<20%), with extremely high spelling accuracy.
- **Ins 0.7%**: Hallucination issues completely eradicated; the model no longer generates repetitive characters.
- **WER 55.4%**: Primary errors come from homophone substitutions (Sub) or deletions due to fast speech (Del), which is acceptable for an agglutinative language.

---

## 🛠️ Installation

### Requirements
- **OS**: Linux (Ubuntu 20.04+ / WSL2)
- **GPU**: NVIDIA GPU (Recommended VRAM 8GB+)
- **Memory**: 16GB+ RAM

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/3rskine/SeediqSpeech-low-resource-asr.git
cd SeediqSpeech-low-resource-asr

# 2. Create Conda Environment
conda create -n espnet_asr python=3.10
conda activate espnet_asr

# 3. Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Install ESPnet
pip install espnet==202401

# 5. Install Key Dependencies (Note transformers version)
pip install transformers==4.46.3 peft loralib tensorboard
```

---

## 🚀 Quick Start

### Full Training Pipeline
```bash
# Includes data prep, feature extraction, LM training, and ASR training
./run.sh \
    --stage 1 \
    --stop_stage 13 \
    --ngpu 1 \
    --train_set train_mixed \
    --valid_set seediq_dev \
    --test_sets "seediq_test" \
    --asr_config conf/train_asr_xlsr_lora.yaml \
    --lm_config conf/train_lm.yaml \
    --token_type char \
    --use_lm true
```

### Inference Only
If you already have pretrained weights:
```bash
./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --ngpu 1 \
    --inference_config conf/decode_asr_lm.yaml \
    --use_lm true
```

---

## 📂 File Structure
```
SeediqSpeech-low-resource-asr/
├── conf/                          
│   ├── train_asr_xlsr_lora.yaml  # ASR Config (LoRA)
│   ├── train_lm.yaml             # LM Config
│   └── decode_asr_lm.yaml        # Decode Config (Beam Search)
├── local/                         # Data prep & crawler scripts
├── data/                         
│   ├── train_mixed/              # Merged training data (CV + Crawler)
│   └── seediq_test/              # Test data
├── exp/                          # Experiments (Logs, Checkpoints)
├── asr.sh                        # Core task script
└── README.md                     
```

---

## 📖 Citation

If you use this code or model in your research, please cite:
```bibtex
@misc{seediqspeech2026,
  author = {Erskine},
  title = {SeediqSpeech: Low-Resource ASR for Seediq Language via XLS-R and LoRA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/3rskine/SeediqSpeech-low-resource-asr}
}
```

---

## 🙏 Acknowledgments
- **Tech Support**: [ESPnet](https://github.com/espnet/espnet), Meta AI (XLS-R), Hugging Face.
- **Data Sources**: 
  - **[Mozilla Common Voice](https://moztw.org/common-voice/)**: Thanks to the MozTW community.
  - **Indigenous Language E-Learning Center**: For providing rich sentence corpora.
  - **NCCU ALCD**: For structured learning materials.

## 📄 License
Apache License 2.0
