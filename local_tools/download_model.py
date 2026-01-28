from espnet_model_zoo.downloader import ModelDownloader
import shutil
import os

print("正在啟動 ESPnet Model Zoo 下載器...")
d = ModelDownloader()

# 指定官方標準 Transformer 模型
tag = "shinji_watanabe/librispeech_asr_train_asr_transformer_raw_en_bpe5000"

print(f"開始下載: {tag}")
# download_and_unpack 會自動處理下載與解壓縮
model_info = d.download_and_unpack(tag)

print(f"下載完成！原始路徑: {model_info['asr_model_file']}")

# 將模型複製到當前目錄並改名為 model.pth
shutil.copy(model_info['asr_model_file'], "model.pth")

size_mb = os.path.getsize("model.pth") / (1024 * 1024)
print(f"成功！模型已就位: model.pth ({size_mb:.2f} MB)")
