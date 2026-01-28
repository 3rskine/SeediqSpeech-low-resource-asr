import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys
import torch

# --- 路徑設定 ---
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"
# 使用平均權重 (更穩定)
lm_model_path = "exp/lm_train_lm_seediq_en_bpe5000/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_en_bpe5000/config.yaml"

# 檢查檔案
if not os.path.exists(lm_model_path):
    print(f"❌ 找不到 LM 模型: {lm_model_path}")
    sys.exit(1)

print(f"🚀 載入模型中...\nASR: {asr_model_path}\nLM:  {lm_model_path}")

try:
    # 初始化 (加入 LM 設定)
    speech2text = Speech2Text(
        asr_train_config=asr_config_path,
        asr_model_file=asr_model_path,
        lm_train_config=lm_config_path,
        lm_file=lm_model_path,
        device="cpu", # 強制 CPU 避免 OOM
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.3,
        lm_weight=0.3, # LM 權重 (關鍵參數)
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
        print(f"\n🎧 正在聽 (With LM): {wav_file}")
        
        nbests = speech2text(speech)
        text, *_ = nbests[0]
        
        print("-" * 30)
        print(f"🗣️  結果: {text}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ 推論失敗: {e}")

if __name__ == "__main__":
    recognize_file("test.wav")
