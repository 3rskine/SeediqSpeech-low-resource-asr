# 🎙️ SeediqSpeech: Low-Resource ASR for Seediq Language

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ESPnet](https://img.shields.io/badge/ESPnet-202401-green.svg)](https://github.com/espnet/espnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

[**中文說明**](README.md) | [**English**](README_en.md)

**賽德克語 (Seediq) 低資源語音辨識系統**

本專案基於 [ESPnet](https://github.com/espnet/espnet) 框架，採用自監督學習預訓練模型 (XLS-R) 結合 **LoRA (Low-Rank Adaptation)** 高效微調技術，並整合 **RNN 語言模型 (RNNLM)**，成功在極低資源下實現台灣原住民語言——賽德克語的高可用性端到端語音辨識。

---

## 📋 目錄

- [專案亮點](#-專案亮點)
- [模型架構](#-模型架構)
- [實驗歷程與優化](#-實驗歷程與優化)
- [最終實驗結果](#-最終實驗結果)
- [環境安裝](#️-環境安裝)
- [快速開始](#-快速開始)
- [引用與致謝](#-引用)

---

## 🌟 專案亮點

- **極低資源突破**：從僅 1 小時 (Common Voice) 擴增至約 4 小時資料 (爬蟲擴充)，成功將 CER 從 >300% 降至 19.5%。
- **幻覺抑制**：透過全大寫正規化 (Uppercase Normalization) 與 16kHz 取樣率修正，解決了低資源模型常見的重複字元幻覺 (Insertion Error 0.7%)。
- **高效訓練**：使用 LoRA 僅訓練 <1% 參數，配合單張 RTX 4060 (8GB) 即可完成訓練。

---

## 🏗️ 模型架構

| 組件 | 詳細資訊 |
|------|---------|
| **Framework** | ESPnet2 (v202401) |
| **Pretrained Model** | `facebook/wav2vec2-xls-r-300m` |
| **Frontend** | S3PRL (Self-Supervised Speech Processing) |
| **Encoder** | XLS-R Transformer (凍結主體，僅 LoRA 微調) |
| **LoRA Config** | Rank=8, Alpha=16, Target: `q_proj`, `v_proj` |
| **Decoder** | 6-layer Transformer (d_model=256, d_ff=1024) |
| **Language Model** | RNNLM (2-layer LSTM, unit=512) |
| **Tokenizer** | Character-based (42 tokens, Uppercase Only) |
| **Optimization** | AdamW (lr=5e-4, warmup=1000 steps) |

---

## 🧪 實驗歷程與優化

本專案經歷了從基線測試到最終優化的完整迭代過程：

### 📅 Phase 0: 基線測試與環境建置 ( - 1/15)
- **嘗試**：使用 Librispeech 預訓練模型進行 Zero-shot 遷移。
- **結果**：**完全失敗 (WER 99.8%)**。確認聲學特徵差異過大，必須進行 Fine-tuning。
- **修正**：解決 HuggingFace `transformers` 版本相容性問題 (降級至 v4.46.3)，並修復 OOM 問題 (調整 `nj=4`)。

### 📅 Phase 1: 初步微調 (1/17 - 1/23)
- **資料**：僅使用 Common Voice (約 1 小時)。
- **結果**：CER 34.7%, WER 82.8%。
- **問題**：模型產生嚴重「幻覺」(Insertion Error 高)，且語言模型 (LM) 因訓練資料 (聖經) 風格單一導致過擬合。
- **優化**：進行 Grid Search 尋找最佳解碼參數 (`lm_weight 0.1`, `penalty 1.0`)，CER 降至 28.8%，但仍受限於資料量瓶頸。

### 📅 Phase 2: 資料擴增與架構重構 (1/24 - 1/28)
- **資料擴增**：撰寫爬蟲抓取「族語 E 樂園」與「政大教材」，資料量從 885 句提升至 **3158 句**。
- **關鍵除錯 (Critical Fixes)**：
    1. **取樣率修正**：排除 48k/16k 混用導致的時間膨脹問題，統一轉檔為 16kHz。
    2. **全大寫正規化 (The Game Changer)**：發現模型對混合大小寫敏感導致 `<unk>` 爆炸。將所有訓練/測試文本強制轉為 **全大寫**。
    3. **ID 對齊**：修復 `wav.scp` 與 `text` 排序不一致導致的訓練崩潰。
    4. **LM 重練**：基於擴增後的標準化文本重練 RNNLM，PPL (困惑度) 從 179.27 大幅降至 **8.77**。

---

## 📊 最終實驗結果 (Phase 2 Final)

在解決大小寫與對齊問題，並加入高品質 LM 後的最終測試集表現：

| 實驗階段 | Config | CER (%) | WER (%) | Ins (%) | 狀態 |
|:-------:|:-------|:-------:|:-------:|:-------:|:----:|
| **Baseline** | Zero-shot | 100.0 | 99.8 | - | ❌ |
| **Phase 1** | 1hr Data (No LM) | 34.7 | 82.8 | 4.3 | ⚠️ |
| **Phase 1++**| 1hr Data + Old LM | 28.8 | 77.6 | Medium | ⚠️ |
| **Phase 2** | **3k Data + New LM** | **19.5** | **55.4** | **0.7** | ✅ |

### 🔍 結果分析
- **CER 19.5%**: 達成 <20% 的高可用性目標，拼寫準確度極高。
- **Ins 0.7%**: 幻覺問題徹底根除，模型不再產生重複字元。
- **WER 55.4%**: 對於黏著語系而言，此數值已屬優秀，主要誤差來自同音異字替換 (Sub) 或語速過快導致的漏字 (Del)。

---

## 🛠️ 環境安裝

### 系統需求
- **OS**: Linux (Ubuntu 20.04+ / WSL2)
- **GPU**: NVIDIA GPU (建議 VRAM 8GB+)
- **Memory**: 16GB+ RAM

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

# 5. 安裝關鍵依賴 (注意 transformers 版本)
pip install transformers==4.46.3 peft loralib tensorboard
```

---

## 🚀 快速開始

### 完整訓練流程
```bash
# 包含資料清洗、特徵提取、LM 訓練與 ASR 訓練
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

### 僅推論 (Inference)
若已有訓練好的模型權重：
```bash
./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --ngpu 1 \
    --inference_config conf/decode_asr_lm.yaml \
    --use_lm true
```

---

## 📂 檔案結構
```
SeediqSpeech-low-resource-asr/
├── conf/                          
│   ├── train_asr_xlsr_lora.yaml  # ASR 訓練設定 (LoRA)
│   ├── train_lm.yaml             # LM 訓練設定
│   └── decode_asr_lm.yaml        # 解碼設定 (Beam Search)
├── local/                         # 資料處理與爬蟲腳本
├── data/                         
│   ├── train_mixed/              # 整合後的訓練資料 (CV + Crawler)
│   └── seediq_test/              # 測試資料
├── exp/                          # 實驗結果 (Log, Checkpoints)
├── asr.sh                        # 核心任務腳本
└── README.md                     
```

---

## 📖 引用

如果本專案對您的研究有幫助，請引用：
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

## 🙏 致謝
- **技術支援**: [ESPnet](https://github.com/espnet/espnet), Meta AI (XLS-R), Hugging Face.
- **資料來源**: 
  - **[Mozilla Common Voice](https://moztw.org/common-voice/)**: 感謝 MozTW 社群推動。
  - **族語 E 樂園**: 提供豐富的句型語料。
  - **國立政治大學原住民族研究中心**: 提供結構化教材資源。

## 📄 License
Apache License 2.0
