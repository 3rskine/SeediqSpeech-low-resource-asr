from espnet_model_zoo.downloader import ModelDownloader
import shutil
import os

# 1. 強制移除可能導致 401 錯誤的環境變數
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

print("正在啟動 ESPnet Model Zoo (已清除 Auth Token)...")
d = ModelDownloader()

# 2. 改用 Kan Bayashi 的模型 (架構完全相同：Transformer)
# 這個連結通常比較穩定
tag = "kan-bayashi/librispeech_asr_train_asr_transformer_raw_en_bpe5000"

print(f"開始下載替代模型: {tag}")
try:
    model_info = d.download_and_unpack(tag)
    print(f"下載完成！原始路徑: {model_info['asr_model_file']}")

    # 3. 複製並改名
    shutil.copy(model_info['asr_model_file'], "model.pth")
    
    size_mb = os.path.getsize("model.pth") / (1024 * 1024)
    print(f"成功！模型已就位: model.pth ({size_mb:.2f} MB)")

except Exception as e:
    print(f"下載失敗: {e}")
    print("嘗試備案：直接使用 wget 下載...")
    # 這是最後的備用連結
    url = "https://zenodo.org/record/4031961/files/asr_train_asr_transformer_raw_en_bpe5000_valid.acc.best.pth?download=1"
    os.system(f"wget -O model.pth '{url}'")
    if os.path.exists("model.pth") and os.path.getsize("model.pth") > 1000:
         print("wget 下載成功！")
    else:
         print("wget 也失敗了，請檢查網路。")
