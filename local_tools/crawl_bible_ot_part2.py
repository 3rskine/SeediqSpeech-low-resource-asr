import requests
from bs4 import BeautifulSoup
import time
import re

# 你指定的書卷與章節數
books = [
    ("EXO", 40), # 出埃及記
    ("LEV", 27), # 利未記
    ("NUM", 36), # 民數記
    ("DEU", 34), # 申命記
    ("JOS", 24)  # 約書亞記
]

base_url = "https://www.bible.com/bible/3109/{}.{}.STGDAYA"
output_file = "seediq_bible_part2.txt"
headers = {"User-Agent": "Mozilla/5.0"}

def clean_text_for_asr(text):
    text = re.sub(r'\d+', '', text)      # 去數字
    text = re.sub(r'[^\w\s]', ' ', text) # 去標點
    text = text.upper()                  # 轉大寫
    return re.sub(r'\s+', ' ', text).strip()

print(f"🚀 開始爬取摩西五經與歷史書 (共 {len(books)} 卷)...")

with open(output_file, "w", encoding="utf-8") as f:
    for book_code, total_chapters in books:
        print(f"\n📘 {book_code} ({total_chapters} 章): ", end="")
        for chapter in range(1, total_chapters + 1):
            url = base_url.format(book_code, chapter)
            try:
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # 抓取內容 (相容性寫法)
                    div = soup.find("div", class_=lambda x: x and "ChapterContent_chapter" in x)
                    if not div: div = soup.find("div", class_="yv-bible-text")
                    
                    if div:
                        cleaned = clean_text_for_asr(div.get_text(separator=" "))
                        if cleaned:
                            f.write(cleaned + "\n")
                            print(".", end="", flush=True)
                        else:
                            print("x", end="", flush=True)
                    else:
                        print("?", end="", flush=True)
                else:
                    print(f"![{resp.status_code}]", end="", flush=True)
            except Exception as e:
                print("E", end="", flush=True)
            
            # 禮貌性延遲
            time.sleep(0.5) 

print(f"\n\n✅ 完成！已儲存至 {output_file}")
