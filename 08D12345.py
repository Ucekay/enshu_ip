# [情報システム工学演習 課題提出用ファイル]

# 説明
# 本ファイルに実装し、`{学籍番号}.py`にリネームの上、必要な入力データ（あれば）とあわせて`{学籍番号}.zip`にまとめること。
# この際、`{学籍番号}.py`を実行したらそのまま読み込める場所に入力データを配置しておくことが望ましい（それが難しい場合はどこかにその旨記載しておくこと）。
# レポートをコンパイルしたもの（`{学籍番号}.pdf`）と合わせてCLEから提出すること。

# ---
# 課題情報
#
# 提出者（以下を書き換え）
submission_id = '08D22047'
# 氏名（以下を書き換え）
submission_name = '木村優介' 

# 概要
# 実装したアプリ・システム・ツールの概要・アピールポイント（簡潔に。詳細はレポートに記入すること）：
submission_highlights = '...'
# 
# 実行に準備（インストールや実行環境）が必要な場合、内容をここに記入。pipで入るものなら、requirements.txtを同梱してくれると更にありがたい：
submission_requirements = '...'
# ---

import cv2
import os
from dotenv import load_dotenv
import requests
import base64
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import re
import csv
import glob

# .envファイルから環境変数を読み込み
load_dotenv()

# Hugging Face API設定
API_URL = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b2_clothes"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}

def query_api(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def decode_mask(mask_string):
    try:
        mask_bytes = base64.b64decode(mask_string)
        mask_image = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_image)
        return mask_array
    except Exception as e:
        print(f"Error decoding mask: {str(e)}")
        raise


def get_clothing_and_accessories_mask(segments):
    target_labels = [
        'Hat',
        'Sunglasses',
        'Upper-clothes',
        'Skirt',
        'Dress',
        'Pants',
        'Belt',
        'Left-shoe',
        'Right-shoe',
        'Bag',
        'Scarf'
    ]
    
    combined_mask = None
    for segment in segments:
        if segment['label'] in target_labels:
            # Base64文字列をデコード
            mask = decode_mask(segment['mask'])
            # マスクが2次元配列であることを確認
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # RGBAの場合は最初のチャンネルを使用
            # バイナリマスクに変換
            mask = mask > 0
            
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.logical_or(combined_mask, mask)
    
    # マスクが生成されなかった場合の処理
    if combined_mask is None:
        print("Warning: No valid segments found for clothing/accessories")
        return np.ones((segments[0]['mask'].shape[0], segments[0]['mask'].shape[1]), dtype=bool)
    
    return combined_mask

def apply_mask_to_image(image, mask):
    rgba_image = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image
    rgba_image[:, :, 3] = 255
    
    if mask is not None:
        rgba_image[~mask, 3] = 0
    
    return rgba_image

def shrink_mask(mask, shrink_pixels=5):
    if mask is None:
        return None
    kernel = np.ones((shrink_pixels, shrink_pixels), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

def process_image(image_path):
    try:
        if not os.getenv('HUGGINGFACE_API_TOKEN'):
            raise ValueError("HUGGINGFACE_API_TOKENが設定されていません。.envファイルを確認してください。")
            
        original_image = np.array(Image.open(image_path))
        segments = query_api(image_path)
        mask = get_clothing_and_accessories_mask(segments)
        shrunken_mask = shrink_mask(mask, shrink_pixels=5)
        masked_image = apply_mask_to_image(original_image, shrunken_mask)
        return masked_image
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

def quantize_colors(image_rgba, max_colors: int = 5, quality: int = 10) -> tuple[list[tuple], list[float]]:
    if image_rgba is None or image_rgba.size == 0 or max_colors < 2 or max_colors > 256:
        return [], []
    
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3] if image_rgba.shape[2] == 4 else np.full(image_rgba.shape[:2], 255, dtype=np.uint8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # BGRからHSVに変換
    image_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = image_hsv.shape[:2]

    # 有効なピクセルのHSV値を収集
    valid_pixels_hsv = []
    for y in range(0, height, quality):
        for x in range(0, width, quality):
            if alpha[y, x] <= 125:  # 透明度が高いピクセルは無視
                continue
            hsv = image_hsv[y, x]
            # OpenCVのHSVスケール（H: 0-179, S: 0-255, V: 0-255）を
            # 標準的なHSVスケール（H: 0-360, S: 0-100, V: 0-100）に変換
            normalized_hsv = [
                float(hsv[0]) * 2.0,         # H: 179 -> 360
                float(hsv[1]) * 100.0 / 255, # S: 255 -> 100
                float(hsv[2]) * 100.0 / 255  # V: 255 -> 100
            ]
            valid_pixels_hsv.append(normalized_hsv)

    if not valid_pixels_hsv:
        return [], []

    valid_pixels_array = np.float32(valid_pixels_hsv)

    # HSV空間でクラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(valid_pixels_array, max_colors, None, criteria, 10,
                                  cv2.KMEANS_RANDOM_CENTERS)

    # HSV値をタプルに変換（標準的なスケールのまま）
    centers_hsv = [(int(hsv[0]), int(hsv[1]), int(hsv[2])) for hsv in centers]
    
    # 各クラスタの割合を計算
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    percentages = [count / total_pixels * 100 for count in counts]
    
    # 割合で降順ソート
    colors_with_percentages = list(zip(centers_hsv, percentages))
    colors_with_percentages.sort(key=lambda x: x[1], reverse=True)
    
    colors, percentages = zip(*colors_with_percentages)
    
    return list(colors), list(percentages)

if __name__ == "__main__":
    results = []
    csv_path = "color_analysis_results.csv"
    
    try:
        # 1から10080までの画像を順番に処理
        for i in range(10080, 10081):
            image_path = f"./datasets/{i}.jpg"
            print(f"Processing {i}/10080: {image_path}...")
            
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found, skipping...")
                continue
            
            # 画像処理
            masked_image = process_image(image_path)
            if masked_image is None:
                raise RuntimeError(f"画像処理に失敗しました: {image_path}")
                
            # 色の抽出
            colors, percentages = quantize_colors(masked_image, max_colors=5, quality=1)
            
            # 結果をフォーマット
            color_data = []
            for color, percentage in zip(colors, percentages):
                color_data.extend([f"({color[0]:.1f},{color[1]:.1f},{color[2]:.1f})", f"{percentage:.2f}"])
            
            # 結果を保存（5色分確保、足りない場合は空のデータで補完）
            while len(color_data) < 10:  # 5色×2(色とパーセント)
                color_data.extend(["", ""])
                
            results.append([i] + color_data)
            
            # 100件ごとに中間保存（追記モード）
            if len(results) >= 100:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(results)
                print(f"Intermediate save at {i}/10080 completed.")
                results = []  # バッファをクリア
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        print("現在までの結果を保存して終了します。")
        
        # エラー発生時に残りのデータを保存
        if results:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)
        
        # エラーを再度投げて処理を完全に終了
        raise
    
    finally:
        # 正常終了時の残りのデータを保存
        if results:  # 未保存のデータがある場合
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)
        
        total_processed = sum(1 for _ in csv.reader(open(csv_path))) - 1  # ヘッダーを除く
        print(f"分析結果を{csv_path}に保存しました。")
        print(f"処理完了: {total_processed}件の画像を処理しました。")
