import requests
from bs4 import BeautifulSoup
import time
import re
import sys

# --- 設定 ---
# 創世紀 (Genesis) 共有 50 章
start_chapter = 1
end_chapter = 50
output_file = "seediq_bible_cleaned.txt"
base_url = "https://www.bible.com/bible/3109/GEN.{}.STGDAYA"

# Headers 偽裝成瀏覽器，避免被擋
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text_for_asr(text):
    """
    將聖經文字轉換為 ASR 訓練格式：
    1. 去除章節數字 (1, 2...)
    2. 去除標點符號
    3. 強制轉大寫 (配合你的 Tokenizer Hack)
    4. 去除多餘空白
    """
    # 移除數字 (例如句首的 "1 ")
    text = re.sub(r'\d+', '', text)
    
    # 移除標點符號 (保留空格和字母)
    # 賽德克語可能有 '- (連字號)，視情況保留或去除，這裡先去除以求保險
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 強制轉大寫
    text = text.upper()
    
    # 縮減多餘空白 (把多個空白變一個)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print(f"🚀 開始爬取賽德克語聖經 (GEN 1-{end_chapter})...")
print(f"💾 目標檔案: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    for chapter in range(start_chapter, end_chapter + 1):
        url = base_url.format(chapter)
        print(f"📖 正在處理第 {chapter} 章...", end="")
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"❌ 失敗 (Status: {response.status_code})")
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Bible.com 的經文通常在 class 包含 'ChapterContent_chapter' 的 div 裡
            # 裡面的每一節通常是 span
            # 最暴力的解法：直接抓所有文字，然後清洗
            content_div = soup.find("div", class_=lambda x: x and "ChapterContent_chapter" in x)
            
            if not content_div:
                # 備用方案：如果改版了，抓所有 class 為 content 的
                content_div = soup.find("div", class_="yv-bible-text")

            if content_div:
                raw_text = content_div.get_text(separator=" ")
                cleaned_line = clean_text_for_asr(raw_text)
                
                # 寫入檔案 (一行一章，或者你可以再切更細)
                # 這裡為了讓 LM 學到更多連貫性，我們把整章寫成一行，或者依據句號切分
                # 既然已經去掉了標點，我們就整章寫入
                if cleaned_line:
                    f.write(cleaned_line + "\n")
                    print(f"✅ 成功 (長度: {len(cleaned_line)})")
                else:
                    print("⚠️ 抓不到內容")
            else:
                print("❌ 找不到經文容器 (網站結構可能改變)")

        except Exception as e:
            print(f"❌ 錯誤: {e}")
        
        # 禮貌性延遲，避免被鎖 IP
        time.sleep(1)

print("-" * 30)
print("🎉 爬取完成！請檢查 seediq_bible_cleaned.txt")
