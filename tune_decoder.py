import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys

# --- 固定路徑 ---
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"
lm_model_path = "exp/lm_train_lm_seediq_en_bpe5000/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_en_bpe5000/config.yaml"
wav_file = "test.wav"

# --- 實驗參數網格 ---
lm_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
ctc_weights = [0.3, 0.5]

print(f"🔬 開始 Grid Search (測試音檔: {wav_file})...")
print("-" * 60)
print(f"{'LM W':<10} | {'CTC W':<10} | {'Result':<40}")
print("-" * 60)

# 讀取音檔
if not os.path.exists(wav_file):
    print("❌ 找不到 test.wav")
    sys.exit(1)
speech, rate = soundfile.read(wav_file)

for ctc_w in ctc_weights:
    for lm_w in lm_weights:
        try:
            # 初始化模型 (每次換參數)
            speech2text = Speech2Text(
                asr_train_config=asr_config_path,
                asr_model_file=asr_model_path,
                lm_train_config=lm_config_path,
                lm_file=lm_model_path,
                device="cpu",
                minlenratio=0.0,
                maxlenratio=0.0,
                ctc_weight=ctc_w,
                lm_weight=lm_w,  # 變數
                beam_size=10,
                batch_size=0
            )
            
            # 推論
            nbests = speech2text(speech)
            text, *_ = nbests[0]
            
            print(f"{lm_w:<10} | {ctc_w:<10} | {text}")
            
        except Exception as e:
            print(f"{lm_w:<10} | {ctc_w:<10} | ❌ Error")

print("-" * 60)
print("✅ 實驗結束。請挑選最接近 'QMI KA DORIQ SU MI DRUNI HAN' 的組合。")
