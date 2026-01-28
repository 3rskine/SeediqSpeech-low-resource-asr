import soundfile
from espnet2.bin.asr_inference import Speech2Text
import os
import sys
import glob
import random

# --- 設定區 ---
# 模型路徑
model_path = "exp/asr_seediq_upper/valid.acc.ave.pth"
# 設定檔路徑
config_path = "exp/asr_seediq_upper/config.yaml"
# 你剛剛複製過來的檔案
user_wav = "test.wav"

if not os.path.exists(model_path):
    print(f"❌ 找不到模型檔案: {model_path}")
    sys.exit(1)

print("🚀 正在載入賽德克語模型 (請稍候)...")
try:
    speech2text = Speech2Text(
        asr_train_config=config_path,
        asr_model_file=model_path,
        device="cpu",  # 為了避免記憶體問題，我們強制用 CPU 跑預測
        minlenratio=0.0,
        maxlenratio=0.0,
        ctc_weight=0.3,
        beam_size=10,
        batch_size=0
    )
    print("✅ 模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    sys.exit(1)

def recognize_file(wav_file):
    if not os.path.exists(wav_file):
        print(f"⚠️ 找不到檔案: {wav_file}")
        return

    # 讀取音檔
    try:
        speech, rate = soundfile.read(wav_file)
        print(f"\n🎧 正在聽: {wav_file} (採樣率: {rate})")
        
        # 進行辨識
        nbests = speech2text(speech)
        text, *_ = nbests[0]
        
        # 加上裝飾線讓結果更明顯
        print("-" * 30)
        print(f"🗣️  辨識結果: {text}")
        print("-" * 30)
        return text
    except Exception as e:
        print(f"❌ 處理檔案時發生錯誤: {e}")

if __name__ == "__main__":
    # 1. 優先測試使用者提供的檔案
    if os.path.exists(user_wav):
        print(f"\n🎯 發現使用者指定檔案，優先測試！")
        recognize_file(user_wav)
    
    # 2. 如果沒有使用者檔案，才去隨機抓測試集的
    else:
        print(f"\n⚠️ 沒看到 {user_wav}，改為隨機抽取測試集檔案...")
        search_path = "dump/raw/wavs_seediq/seediq_test/*.wav"
        wav_files = glob.glob(search_path)
        
        if not wav_files:
            print(f"❌ 在 {search_path} 找不到任何測試音檔")
        else:
            # 隨機挑 1 個檔案來測試
            test_wav = random.choice(wav_files)
            recognize_file(test_wav)
