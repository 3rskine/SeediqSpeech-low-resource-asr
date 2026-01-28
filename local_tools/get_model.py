import shutil
import os
from espnet_model_zoo.downloader import ModelDownloader

# 這是我們剛剛用過的模型標籤
tag = "espnet/shinji_watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

print(f"正在搜尋快取中的模型: {tag} ...")
d = ModelDownloader()
# 這會直接回傳快取中的路徑 (因為之前下載過了)
data = d.download_and_unpack(tag)
src_path = data["asr_model_file"]

print(f"找到模型檔案: {src_path}")

# 建立目標資料夾
target_dir = "exp/pretrained_model"
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, "valid.acc.best.pth")

# 複製檔案
shutil.copy(src_path, target_path)
print(f"成功複製模型到: {target_path}")
