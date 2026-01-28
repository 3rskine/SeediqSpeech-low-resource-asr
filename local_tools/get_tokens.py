import shutil
import os
from espnet_model_zoo.downloader import ModelDownloader

print("正在尋找 tokens.txt ...")
d = ModelDownloader()
# 取得模型路徑
data = d.download_and_unpack("espnet/shinji_watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best")
model_file = data["asr_model_file"]

# 回推到 snapshot 目錄
snapshot_root = os.path.dirname(os.path.dirname(model_file))

# 暴力搜尋 tokens.txt
found = False
target_path = "data/en_token_list/bpe_unigram5000/tokens.txt"

for root, dirs, files in os.walk(snapshot_root):
    if "tokens.txt" in files:
        # 為了保險起見，我們要找的是跟 bpe.model 在一起的那個
        if "bpe_unigram5000" in root: 
            src_path = os.path.join(root, "tokens.txt")
            print(f"✅ 找到了！: {src_path}")
            shutil.copy(src_path, target_path)
            print(f"-> 已複製到: {target_path}")
            found = True
            break

if not found:
    print("❌ 找不到 tokens.txt，請檢查模型快取。")
else:
    print("字典列表準備完成！")
