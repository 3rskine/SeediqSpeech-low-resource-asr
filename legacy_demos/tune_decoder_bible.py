import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys

# --- 路徑設定 (聖經版) ---
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"
lm_model_path = "exp/lm_train_lm_seediq_bible_en_bpe5000/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_bible_en_bpe5000/config.yaml"
wav_file = "test.wav"

# 防呆
if not os.path.exists(lm_model_path):
    lm_model_path = "exp/lm_train_lm_seediq_bible_en_bpe5000/valid.loss.min.pth"

# --- 擴大搜索範圍 ---
# 既然 LM 變強了，我們可以嘗試稍微高一點的 LM 權重，或者更強的 CTC
lm_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ctc_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print(f"🔬 開始 Grid Search (Model: Bible LM)...")
print("-" * 70)
print(f"{'LM':<5} | {'CTC':<5} | {'Result':<40}")
print("-" * 70)

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
        except:
            pass

print("-" * 70)
