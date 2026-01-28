import os
import shutil

# 設定閾值 (字元數)
MAX_LEN = 400
DATA_DIR = "data/train_mixed"
BACKUP_DIR = "data/train_mixed_backup"

def main():
    # 1. 備份資料 (這很重要！)
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(DATA_DIR, BACKUP_DIR)
    print(f"📦 已備份原始資料至: {BACKUP_DIR}")

    # 2. 讀取 text 檔並篩選
    valid_ids = set()
    kept_lines = []
    removed_count = 0
    
    with open(os.path.join(DATA_DIR, "text"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            utt_id = parts[0]
            content = " ".join(parts[1:])
            
            # 檢查長度
            if len(content) <= MAX_LEN:
                valid_ids.add(utt_id)
                kept_lines.append(line)
            else:
                removed_count += 1
                print(f"❌ 移除過長句子 ({len(content)}): {utt_id}")

    # 3. 寫回 text 檔
    with open(os.path.join(DATA_DIR, "text"), "w") as f:
        f.writelines(kept_lines)

    # 4. 同步過濾 wav.scp, utt2spk, spk2utt
    for filename in ["wav.scp", "utt2spk", "spk2utt"]:
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

    print(f"\n🧹 清洗完成！")
    print(f"❌ 共移除: {removed_count} 句")
    print(f"✅ 剩餘: {len(valid_ids)} 句")

if __name__ == "__main__":
    main()
