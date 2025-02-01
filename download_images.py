import csv
import os
import requests
from pathlib import Path
import time

# ダウンロード先ディレクトリの作成
DOWNLOAD_DIR = Path('./datasets')
DOWNLOAD_DIR.mkdir(exist_ok=True)

def download_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# CSVファイルの読み込みと画像のダウンロード
with open('wear_rankings.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for i, row in enumerate(csv_reader, 1):
        ranking = row['Ranking']
        url = row['URL']
        filename = DOWNLOAD_DIR / f"{ranking}.jpg"
        download_image(url, filename)
        
        # 120枚ごとに5秒待機
        if i % 120 == 0:
            print(f"Downloaded {i} images. Waiting for 5 seconds...")
            time.sleep(5)
