import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys

# 路徑
asr_model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
asr_config_path = "exp/asr_seediq_upper/config.yaml"
lm_model_path = "exp/lm_train_lm_seediq_final_v3/valid.loss.ave.pth"
lm_config_path = "exp/lm_train_lm_seediq_final_v3/config.yaml"

if not os.path.exists(lm_model_path):
    lm_model_path = "exp/lm_train_lm_seediq_final_v3/valid.loss.min.pth"

print(f"🚀 執行最終高寬度搜索 (High Beam Search)...")

speech2text = Speech2Text(
    asr_train_config=asr_config_path,
    asr_model_file=asr_model_path,
    lm_train_config=lm_config_path,
    lm_file=lm_model_path,
    device="cpu",
    ctc_weight=0.5,   # 鎖定最佳
    lm_weight=0.2,    # 鎖定最佳
    beam_size=60,     # <--- 暴力全開！
    batch_size=0
)

wav_file = "test.wav"
speech, rate = soundfile.read(wav_file)
nbests = speech2text(speech)
text, *_ = nbests[0]

print("-" * 30)
print(f"Beam=60 結果: {text}")
print("-" * 30)
