import os
import pandas as pd
from tqdm import tqdm
import subprocess

# --- 設定區 ---
SOURCE_ROOT = "cv-corpus-24.0-2025-12-05/trv"
OUTPUT_ROOT = "dump/raw"
TARGET_SR = 16000

splits = {
    "train": "seediq_train",
    "dev":   "seediq_dev",
    "test":  "seediq_test"
}

def process_split(split_name, output_name):
    tsv_path = os.path.join(SOURCE_ROOT, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        print(f"❌ 找不到檔案: {tsv_path}")
        return

    save_dir = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(tsv_path, sep="\t", on_bad_lines='skip')
    except:
        df = pd.read_csv(tsv_path, sep="\t", error_bad_lines=False)

    print(f"🚀 正在處理 {split_name} -> {output_name} (共 {len(df)} 筆)...")

    wav_scp_path = os.path.join(save_dir, "wav.scp")
    text_path = os.path.join(save_dir, "text")
    utt2spk_path = os.path.join(save_dir, "utt2spk")

    with open(wav_scp_path, "w", encoding="utf-8") as f_wav, \
         open(text_path, "w", encoding="utf-8") as f_text, \
         open(utt2spk_path, "w", encoding="utf-8") as f_utt:
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                client_id = str(row['client_id'])
                path = str(row['path'])
                sentence = str(row['sentence'])
                
                mp3_path = os.path.join(SOURCE_ROOT, "clips", path)
                wav_storage = os.path.join(OUTPUT_ROOT, "wavs_seediq", split_name)
                os.makedirs(wav_storage, exist_ok=True)
                
                wav_filename = path.replace(".mp3", ".wav")
                wav_path = os.path.join(wav_storage, wav_filename)
                
                # 這次不檢查 wav 是否存在，因為 wav 已經在上次轉好了，我們主要目的是「重寫 text 檔」
                # 但為了保險，如果 wav 不見了還是轉一下
                if not os.path.exists(wav_path):
                    cmd = f"ffmpeg -i {mp3_path} -ar {TARGET_SR} -ac 1 {wav_path} -v error -y"
                    os.system(cmd)

                if len(client_id) > 8:
                    spk_id = f"spk_{client_id[:8]}"
                else:
                    spk_id = f"spk_{client_id}"
                utt_id = f"{spk_id}_{wav_filename.replace('.wav', '')}"

                f_wav.write(f"{utt_id} {os.path.abspath(wav_path)}\n")
                
                # ★★★ 關鍵修改：轉成大寫！ ★★★
                clean_text = sentence.upper().strip()
                f_text.write(f"{utt_id} {clean_text}\n")
                
                f_utt.write(f"{utt_id} {spk_id}\n")
            except:
                continue

    print(f"✅ {output_name} (大寫版) 處理完成！")

if __name__ == "__main__":
    for split, out_name in splits.items():
        process_split(split, out_name)
    print("\n🎉 資料修正完成！現在是 ALL CAPS 模式！")
