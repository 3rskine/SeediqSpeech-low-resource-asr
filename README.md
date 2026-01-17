# 🎙️ SeediqSpeech: Low-Resource ASR for Seediq Language

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ESPnet](https://img.shields.io/badge/ESPnet-202401-green.svg)](https://github.com/espnet/espnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**賽德克語 (Seediq) 低資源語音辨識系統**

本專案基於 [ESPnet](https://github.com/espnet/espnet) 框架,採用自監督學習預訓練模型 (XLS-R) 結合 LoRA 高效微調技術,實現台灣原住民語言——賽德克語的端到端語音辨識。

---

## 📋 目錄

- [模型架構](#-模型架構)
- [實驗結果](#-實驗結果)
- [環境安裝](#️-環境安裝)
- [快速開始](#-快速開始)
- [檔案結構](#-檔案結構)
- [引用](#-引用)
- [致謝](#-致謝)

---

## 🏗️ 模型架構

| 組件 | 詳細資訊 |
|------|---------|
| **Framework** | ESPnet2 |
| **Pretrained Model** | `facebook/wav2vec2-xls-r-300m` |
| **Frontend** | S3PRL (Self-Supervised Speech Processing) |
| **Encoder** | XLS-R Transformer (凍結,僅 LoRA 微調) |
| **LoRA Config** | Rank=8, Alpha=16, Target: `q_proj`, `v_proj` |
| **Decoder** | 6-layer Transformer (d_model=256, d_ff=1024) |
| **Tokenizer** | Character-based (58 tokens) |
| **Optimization** | AdamW (lr=1e-4, warmup=15000 steps) |

### 💡 關鍵技術特點

- ✅ **高效微調**: 使用 LoRA 僅訓練 0.5% 參數量
- ✅ **低資源友好**: 適合單張消費級 GPU (RTX 3090/4090)
- ✅ **端到端訓練**: CTC/Attention 混合損失
- ✅ **字元級解碼**: 避免大型詞彙表,適應形態豐富的黏著語

---

## 📊 實驗結果

基於賽德克語測試集的初步評估結果:

| Dataset | CER (%) | WER (%) | Ins (%) | Del (%) | Sub (%) |
|---------|---------|---------|---------|---------|---------|
| **Dev**  | **24.6** | 81.8 | 3.3 | 8.9 | 12.4 |
| **Test** | **34.7** | 82.8 | 4.3 | 22.9 | 6.9 |

### 🔍 結果分析

- **CER < 35%**: 模型成功學習賽德克語拼寫規則
- **WER 偏高**: 黏著語特性導致,可透過加入語言模型 (n-gram LM) 改善
- **Del 較高**: 建議調整 CTC weight 或增加訓練數據

> 💡 **改進方向**: 
> 1. 加入語言模型 (Stage 7-9)
> 2. 調整 decode beam size
> 3. 數據增強 (Speed perturbation)

---

## 🛠️ 環境安裝

### 系統需求

- **OS**: Linux (Ubuntu 20.04+ / WSL2)
- **GPU**: NVIDIA GPU with CUDA 12.1+
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ free space

### 安裝步驟
```bash
# 1. Clone repository
git clone https://github.com/3rskine/SeediqSpeech-low-resource-asr.git
cd SeediqSpeech-low-resource-asr

# 2. 建立 Conda 環境
conda create -n espnet_asr python=3.10
conda activate espnet_asr

# 3. 安裝 PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. 安裝 ESPnet
pip install espnet==202401

# 5. 安裝其他依賴
pip install transformers peft loralib tensorboard
```

---

## 🚀 快速開始

### 訓練流程
```bash
# 完整訓練 (Stage 1-13)
./run.sh \
    --stage 1 \
    --stop_stage 13 \
    --ngpu 1 \
    --train_set seediq_train \
    --valid_set seediq_dev \
    --test_sets "seediq_dev seediq_test" \
    --asr_config conf/train_asr_xlsr_lora.yaml \
    --token_type char
```

### 僅推論 (已有模型)
```bash
# Stage 12-13: 解碼與評分
./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --ngpu 1 \
    --inference_config conf/decode_asr.yaml \
    --asr_tag "seediq_xlsr_lora_mvp" \
    --use_lm false
```

### 單檔測試
```python
# 使用訓練好的模型進行推論
python run_inference.py \
    --model_path exp/asr_seediq_xlsr_lora_mvp/valid.acc.ave.pth \
    --audio_path test.wav
```

---

## 📂 檔案結構
```
SeediqSpeech-low-resource-asr/
├── conf/                          # 配置檔案
│   ├── train_asr_xlsr_lora.yaml  # 訓練設定
│   └── decode_asr.yaml           # 解碼設定
├── local/                         # 資料處理腳本
│   └── prep_seediq.py            # 資料預處理
├── pyscripts/                     # Python 工具
├── scripts/                       # Shell 工具
├── utils/                         # Kaldi 工具
├── steps/                         # 訓練步驟
├── run.sh                         # 主執行腳本
├── asr.sh                         # ASR 任務腳本
├── path.sh                        # 環境設定
├── cmd.sh                         # 運算資源配置
├── crawl_bible.py                 # 語料爬蟲
├── demo_*.py                      # 演示腳本
└── README.md                      # 本檔案
```

---

## ⚠️ 資料說明

為節省空間與版權考量,本 Repository **不包含**:

- ❌ `dump/` - 處理後的特徵檔
- ❌ `exp/` - 訓練好的模型權重
- ❌ `data/wav_16k/` - 原始音訊檔案

**如需完整訓練**,請自行準備賽德克語音訊資料並放置於正確目錄。

---

## 📖 引用

如果本專案對您的研究有幫助,請引用:
```bibtex
@misc{seediqspeech2026,
  author = {Your Name},
  title = {SeediqSpeech: Low-Resource ASR for Seediq Language},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/3rskine/SeediqSpeech-low-resource-asr}
}
```

---

## 🙏 致謝
- [ESPnet](https://github.com/espnet/espnet) - 端到端語音處理工具包  
- [Meta AI](https://ai.facebook.com/) - XLS-R 預訓練模型  
- [Hugging Face](https://huggingface.co/) - Transformers 函式庫  
- [Microsoft](https://github.com/microsoft/LoRA) - LoRA 論文與實作  
- **[Mozilla Common Voice 族語錄音補助計畫](https://moztw.org/common-voice/)**  
  感謝 Mozilla 台灣社群（MozTW）與台灣維基媒體協會協力推動的台灣原住民族語開放語音資料庫計畫，提供 CC0 公眾授權的賽德克語語音資料集。

## 📄 License

Apache License 2.0

---

**Made with ❤️ for Indigenous Language Preservation**
