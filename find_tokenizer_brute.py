import os
import shutil
from espnet_model_zoo.downloader import ModelDownloader

print("開始全硬碟搜索 bpe.model ...")

# 1. 取得快取根目錄位置
d = ModelDownloader()
# 下載/確認模型已存在 (這會回傳模型資訊)
data = d.download_and_unpack("espnet/shinji_watanabe_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best")
model_file = data["asr_model_file"]

# 2. 推算出快取的"最上層"目錄 (通常是 snapshots 的上一層)
# 例如: .../espnet_model_zoo/models--espnet.../snapshots/HASH/exp/...
# 我們要回到 .../espnet_model_zoo/
cache_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(model_file))))
print(f"搜索範圍: {cache_root}")

found = False
target_path = "data/en_token_list/bpe_unigram5000/bpe.model"

# 3. 遞迴搜索所有子目錄
for root, dirs, files in os.walk(cache_root):
    if "bpe.model" in files:
        src_path = os.path.join(root, "bpe.model")
        print(f"✅ 找到了！位置在: {src_path}")
        
        # 複製檔案
        shutil.copy(src_path, target_path)
        print(f"-> 已成功複製到: {target_path}")
        found = True
        break # 找到一個就夠了

if not found:
    print("❌ 悲劇：真的找不到 bpe.model。")
    print("這代表 Inference 當時可能使用的是內建配置而非獨立檔案。")
else:
    print("任務完成！可以開始訓練了。")
