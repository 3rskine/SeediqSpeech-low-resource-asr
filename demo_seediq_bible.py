import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys
import torch

# --- 路徑設定 ---
# 1. 聲學模型 (ASR) - 不變
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"

# 2. 語言模型 (LM) - 換成聖經版！
lm_model_path = "exp/lm_train_lm_seediq_bible_en_bpe5000/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_bible_en_bpe5000/config.yaml"

# 防呆檢查
if not os.path.exists(lm_model_path):
    # 嘗試找 best (有些版本存成 min.pth)
    fallback = "exp/lm_train_lm_seediq_bible_en_bpe5000/valid.loss.min.pth"
    if os.path.exists(fallback):
        lm_model_path = fallback
    else:
        print(f"❌ 找不到聖經版 LM 模型: {lm_model_path}")
        sys.exit(1)

print(f"🚀 載入聖經加強版模型...\nASR: {asr_model_path}\nLM:  {lm_model_path}")

try:
    # 初始化
    speech2text = Speech2Text(
        asr_train_config=asr_config_path,
        asr_model_file=asr_model_path,
        lm_train_config=lm_config_path,
        lm_file=lm_model_path,
        device="cpu",
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.5,    # 冠軍參數: 強聲學
        lm_weight=0.2,     # 冠軍參數: 弱語言輔助
        beam_size=10,
        batch_size=0
    )
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    sys.exit(1)

def recognize_file(wav_file):
    if not os.path.exists(wav_file):
        print(f"❌ 找不到音檔: {wav_file}")
        return

    try:
        speech, rate = soundfile.read(wav_file)
        print(f"\n🎧 正在聽 (With Bible LM): {wav_file}")
        
        nbests = speech2text(speech)
        text, *_ = nbests[0]
        
        print("-" * 30)
        print(f"🗣️  結果: {text}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ 推論失敗: {e}")

if __name__ == "__main__":
    recognize_file("test.wav")
