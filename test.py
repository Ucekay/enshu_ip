import cv2
import numpy as np


def quantize_colors(image_bgra, max_colors: int = 5, quality: int = 10) -> tuple[list[tuple], list[float]]:
    if image_bgra is None or image_bgra.size == 0 or max_colors < 2 or max_colors > 256:
        return [], []
    
    bgr = image_bgra[:, :, :3]
    alpha = image_bgra[:, :, 3] if image_bgra.shape[2] == 4 else np.full(image_bgra.shape[:2], 255, dtype=np.uint8)

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
                                  cv2.KMEANS_PP_CENTERS)

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

  # 画像の読み込み
  image_path = "./clothing_and_accessories_masked.png"  # 実際のパスに変更してください
  img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  
  # アルファチャンネルがない場合は追加
  if len(img.shape) == 3 and img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
  
  # 色の分析
  colors, percentages = quantize_colors(img)
  
  # 結果の表示
  for color, percentage in zip(colors, percentages):
    print(f"Color HSV: {color}, Percentage: {percentage:.2f}%")