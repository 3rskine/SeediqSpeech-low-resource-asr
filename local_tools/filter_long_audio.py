import os
import subprocess
import shutil
from tqdm import tqdm

# 設定閾值 (秒)
# 建議設為 20 秒，這是 8GB VRAM/RAM 的安全區
MAX_DURATION = 20.0 

DATA_DIR = "data/train_mixed"
BACKUP_DIR = "data/train_mixed_backup_audio"
WAV_SCP = os.path.join(DATA_DIR, "wav.scp")

def get_duration(file_path):
    """使用 ffprobe 獲取音檔秒數"""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return 999.0 # 讀取失敗就當作有問題，過濾掉

def main():
    # 1. 備份
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(DATA_DIR, BACKUP_DIR)
    print(f"📦 已備份至: {BACKUP_DIR}")

    # 2. 掃描 wav.scp
    print(f"🔍 開始掃描音檔長度 (閾值: {MAX_DURATION}秒)...")
    
    valid_ids = set()
    kept_lines = []
    removed_count = 0
    max_len_found = 0.0
    
    with open(WAV_SCP, "r") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        parts = line.strip().split()
        utt_id = parts[0]
        path = parts[1]
        
        duration = get_duration(path)
        
        if duration > max_len_found:
            max_len_found = duration
            
        if duration <= MAX_DURATION:
            valid_ids.add(utt_id)
            kept_lines.append(line)
        else:
            removed_count += 1
            # print(f"❌ 移除長音檔 ({duration:.2f}s): {utt_id}")

    # 3. 寫回 wav.scp
    with open(WAV_SCP, "w") as f:
        f.writelines(kept_lines)
        
    # 4. 同步過濾 text, utt2spk, spk2utt
    for filename in ["text", "utt2spk", "spk2utt"]:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath): continue
        
        new_content = []
        with open(filepath, "r") as f:
            for line in f:
                utt_id = line.strip().split()[0]
                if utt_id in valid_ids:
                    new_content.append(line)
        
        with open(filepath, "w") as f:
            f.writelines(new_content)

    print(f"\n🧹 過濾完成！")
    print(f"📏 發現最長檔案: {max_len_found:.2f} 秒")
    print(f"❌ 移除過長檔案: {removed_count} 個")
    print(f"✅ 剩餘檔案: {len(valid_ids)} 個")

if __name__ == "__main__":
    main()
