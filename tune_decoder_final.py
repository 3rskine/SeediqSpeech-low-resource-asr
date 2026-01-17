import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys

# --- 路徑設定 (最終版 V3) ---
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"
lm_model_path = "exp/lm_train_lm_seediq_final_v3/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_final_v3/config.yaml"
wav_file = "test.wav"

# 防呆
if not os.path.exists(lm_model_path):
    # 有時候 espnet 會存成 min.pth
    lm_model_path = "exp/lm_train_lm_seediq_final_v3/valid.loss.min.pth"

# --- 參數網格 (Grid) ---
# 這次資料量翻倍，我們可以試試看更相信 LM
lm_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ctc_weights = [0.3, 0.4, 0.5, 0.6, 0.7]

print(f"🔬 開始 Grid Search (Model: Final V3 - Pentateuch+)...")
print("-" * 70)
print(f"{'LM':<5} | {'CTC':<5} | {'Result':<40}")
print("-" * 70)

if not os.path.exists(wav_file):
    print("❌ 找不到 test.wav")
    sys.exit(1)

try:
    speech, rate = soundfile.read(wav_file)
    
    for ctc_w in ctc_weights:
        for lm_w in lm_weights:
            try:
                speech2text = Speech2Text(
                    asr_train_config=asr_config_path,
                    asr_model_file=asr_model_path,
                    lm_train_config=lm_config_path,
                    lm_file=lm_model_path,
                    device="cpu",
                    ctc_weight=ctc_w,
                    lm_weight=lm_w,
                    beam_size=10,
                    batch_size=0
                )
                nbests = speech2text(speech)
                text, *_ = nbests[0]
                print(f"{lm_w:<5} | {ctc_w:<5} | {text}")
            except Exception as e:
                pass
except Exception as e:
    print(f"❌ 發生錯誤: {e}")

print("-" * 70)
