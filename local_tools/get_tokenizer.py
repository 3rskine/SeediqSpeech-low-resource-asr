import shutil
import os
import glob
from espnet_model_zoo.downloader import ModelDownloader

# 設定模型標籤
tag = "espnet/shinji_watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

print(f"正在分析模型包裹: {tag} ...")
d = ModelDownloader()
# 取得模型資訊
data = d.download_and_unpack(tag)

# 取得模型所在的根目錄 (Snapshot root)
# 我們從 model file 的路徑往回推兩層
model_path = data["asr_model_file"]
snapshot_root = os.path.dirname(os.path.dirname(model_path))

print(f"模型根目錄: {snapshot_root}")

# 在目錄中搜尋 bpe.model
found_bpe = False
for root, dirs, files in os.walk(snapshot_root):
    for file in files:
        if file.endswith("bpe.model"):
            src_file = os.path.join(root, file)
            print(f"找到字典檔: {src_file}")
            
            # 複製到目標位置
            dst_file = "data/en_token_list/bpe_unigram5000/bpe.model"
            shutil.copy(src_file, dst_file)
            print(f"-> 已複製到: {dst_file}")
            found_bpe = True
            break
    if found_bpe:
        break

if not found_bpe:
    print("錯誤: 在快取中找不到 bpe.model！")
else:
    print("字典提取成功！")
