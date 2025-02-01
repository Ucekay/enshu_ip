import requests
from bs4 import BeautifulSoup
import time
import csv

all_image_urls = []
for page in range(1, 85):
  url = f"https://wear.jp/men-coordinate/?pageno={page}&user_type=2"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'lxml')
  image_elements = soup.find_all("img", alt=lambda x: x and "さん" in x)
  page_urls = [img.get('src') for img in image_elements]
  all_image_urls.extend(page_urls)

  if page < 84:
    time.sleep(300)  # 5-minute delay

with open('wear_rankings.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['Ranking', 'URL'])
  for idx, url in enumerate(all_image_urls, 1):
    writer.writerow([idx, url])