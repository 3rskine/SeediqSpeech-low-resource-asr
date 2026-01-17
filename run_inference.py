import soundfile
from espnet2.bin.asr_inference import Speech2Text
import torch

# 1. 設定官方神級模型
model_tag = "espnet/shinji_watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

print(f"正在載入模型: {model_tag} ...")
# 第一次執行會自動下載模型 (約 500MB)
speech2text = Speech2Text.from_pretrained(
    model_tag,
    device="cuda",
    beam_size=10,
    batch_size=1
)

# 2. 讀取音檔 (這次是 flac)
audio_file = "test.flac"
print(f"正在讀取音檔: {audio_file} ...")
speech, rate = soundfile.read(audio_file)

# 3. 進行推論
print("正在辨識中 (Inference)...")
nbests = speech2text(speech)
text, *_ = nbests[0]

# 4. 顯示結果
print("=" * 50)
print(f"辨識結果: {text}")
print("=" * 50)
