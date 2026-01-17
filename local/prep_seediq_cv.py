import os
import csv
import sys

# ================= 設定區 (修正版) =================
# 關鍵修正：路徑末端加入了 /trv
CV_ROOT = "/root/espnet/egs2/seediq/cv-corpus-24.0-2025-12-05/trv"
CLIPS_DIR = os.path.join(CV_ROOT, "clips")

DATA_MAP = {
    "train.tsv": "data/seediq_train",
    "dev.tsv":   "data/seediq_dev",
    "test.tsv":  "data/seediq_test"
}
# =================================================

# 再次除錯確認
if os.path.exists(CV_ROOT):
    print(f"DEBUG: 目標資料夾存在: {CV_ROOT}")
    files = os.listdir(CV_ROOT)
    if "train.tsv" in files:
        print("DEBUG: 成功找到 train.tsv！")
    else:
        print(f"DEBUG: 警告！在 {CV_ROOT} 裡面還是沒看到 train.tsv，看到的檔案有: {files[:5]}")
else:
    print(f"DEBUG: 嚴重錯誤！路徑不存在: {CV_ROOT}")
    sys.exit(1)

def process_dataset(tsv_file, output_dir):
    tsv_path = os.path.join(CV_ROOT, tsv_file)
    
    if not os.path.exists(tsv_path):
        print(f"[Warning] 找不到檔案: {tsv_path}，跳過。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    wav_scp = []
    text = []
    utt2spk = []

    print(f"正在處理 {tsv_file} -> {output_dir} ...")
    
    # 讀取 TSV
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter='\t')
        count = 0
        for row in reader:
            filename = row.get('path', '')
            client_id = row.get('client_id', '')
            sentence = row.get('sentence', '')
            
            # 過濾空檔名
            if not filename: continue
            
            # 過濾 Windows Zone.Identifier 垃圾檔 (防呆)
            if "Zone.Identifier" in filename: continue

            utt_id = os.path.splitext(filename)[0]
            
            # 檢查音檔是否存在 (使用絕對路徑)
            abs_path = os.path.join(CLIPS_DIR, filename)
            
            # 這裡我們不做 os.path.exists 檢查，因為檔名可能有 Identifier 干擾，
            # 我們直接信任 TSV 裡的檔名，交給 ffmpeg 去處理。
            
            # 構建 wav.scp (MP3 -> WAV pipe)
            wav_entry = f"ffmpeg -i {abs_path} -f wav -ac 1 -ar 16000 - |"
            
            wav_scp.append((utt_id, wav_entry))
            text.append((utt_id, sentence))
            
            # 處理 speaker id
            spk_id = client_id if client_id else utt_id
            # 替換掉可能導致錯誤的特殊字元
            spk_id = spk_id.replace(" ", "_") 
            utt2spk.append((utt_id, spk_id))
            count += 1

    # 排序 (Kaldi 必要)
    wav_scp.sort()
    text.sort()
    utt2spk.sort()

    # 寫入檔案
    with open(os.path.join(output_dir, 'wav.scp'), 'w') as f:
        for u, entry in wav_scp:
            f.write(f"{u} {entry}\n")
            
    with open(os.path.join(output_dir, 'text'), 'w') as f:
        for u, txt in text:
            f.write(f"{u} {txt}\n")
            
    with open(os.path.join(output_dir, 'utt2spk'), 'w') as f:
        for u, s in utt2spk:
            f.write(f"{u} {s}\n")
            
    print(f"完成！在 {output_dir} 生成了 {count} 筆資料。")

# 執行
for tsv, out_dir in DATA_MAP.items():
    process_dataset(tsv, out_dir)
