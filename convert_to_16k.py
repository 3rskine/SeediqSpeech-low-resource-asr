import os
import subprocess
from tqdm import tqdm

# 設定路徑
INPUT_SCP = "data/train_mixed/wav.scp"
OUTPUT_DIR = "data/wav_16k"
NEW_SCP = "data/train_mixed/wav.scp.new"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 讀取舊的索引
    with open(INPUT_SCP, "r") as f:
        lines = f.readlines()
        
    print(f"🚀 開始轉檔: {len(lines)} 個檔案 -> 16kHz PCM WAV")
    print(f"📂 輸出位置: {os.path.abspath(OUTPUT_DIR)}")
    
    new_lines = []
    
    # 使用 tqdm 顯示進度條
    for line in tqdm(lines):
        parts = line.strip().split()
        utt_id = parts[0]
        src_path = parts[1]
        
        # 定義新檔案路徑
        # 檔名保持 ID 結構，避免重複
        # 例如: Tgdaya_S01_L01_001.wav
        new_filename = f"{utt_id}.wav"
        dst_path = os.path.join(OUTPUT_DIR, new_filename)
        absolute_dst_path = os.path.abspath(dst_path)
        
        # 如果檔案已存在且大小正常，跳過 (支援斷點續傳)
        if not (os.path.exists(absolute_dst_path) and os.path.getsize(absolute_dst_path) > 1000):
            # 呼叫 ffmpeg 轉檔
            # -i 輸入
            # -ar 16000 (設定採樣率 16k)
            # -ac 1 (單聲道)
            # -y (強制覆蓋)
            # -loglevel error (安靜模式)
            cmd = [
                "ffmpeg",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                "-y",
                "-loglevel", "error",
                absolute_dst_path
            ]
            subprocess.run(cmd, check=True)
            
        # 記錄新的對應關係
        new_lines.append(f"{utt_id} {absolute_dst_path}\n")
        
    # 寫入新的 scp 檔
    with open(NEW_SCP, "w") as f:
        f.writelines(new_lines)
        
    print("✅ 轉檔完成！")
    print(f"📝 新索引已建立: {NEW_SCP}")

if __name__ == "__main__":
    main()
