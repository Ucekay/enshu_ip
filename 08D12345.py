# [情報システム工学演習 課題提出用ファイル]

# 説明
# 本ファイルに実装し、`{学籍番号}.py`にリネームの上、必要な入力データ（あれば）とあわせて`{学籍番号}.zip`にまとめること。
# この際、`{学籍番号}.py`を実行したらそのまま読み込める場所に入力データを配置しておくことが望ましい（それが難しい場合はどこかにその旨記載しておくこと）。
# レポートをコンパイルしたもの（`{学籍番号}.pdf`）と合わせてCLEから提出すること。

# ---
# 課題情報
#
# 提出者（以下を書き換え）
submission_id = '08D12345'
# 氏名（以下を書き換え）
submission_name = '大倉史生' 

# 概要
# 実装したアプリ・システム・ツールの概要・アピールポイント（簡潔に。詳細はレポートに記入すること）：
submission_highlights = '...'
# 
# 実行に準備（インストールや実行環境）が必要な場合、内容をここに記入。pipで入るものなら、requirements.txtを同梱してくれると更にありがたい：
submission_requirements = '...'
# ---

import cv2
import numpy as np  # PythonのOpenCVでは、画像はnumpyのarrayとして管理される
from PIL import Image
import matplotlib.pyplot as plt

# 画像ファイルの読み込み・書き込み
src = cv2.imread("sample/sample.jpg")
# img = cv2.imread("sample/sample.jpg",cv2.IMREAD_GRAYSCALE)  # 強制的にグレースケール（白黒画像）として読み込む場合

# てきとうな処理（リサイズしてみた）
#dst = cv2.resize(img,dsize=None,fx=0.5,fy=0.5)  # 縦横半分
dst = cv2.resize(src,dsize=(512,256))  # 指定したサイズ

# 画像の書き込み
# cv2.imwrite("out.png",dst)

# 新しいウインドウを開いて表示する（ウインドウを閉じるか、なにかキーを押すと終了）
cv2.namedWindow('src') # 指定されたタイトルのウインドウを開く
cv2.imshow('src', src)  # 指定されたタイトルのウインドウに画像を表示

cv2.namedWindow('dst') # 指定されたタイトルのウインドウを開く
cv2.imshow('dst', dst)  # 指定されたタイトルのウインドウに画像を表示

cv2.waitKey(0)           # キーが押されるまで{引数}[ms]の間待つ（0の場合はずっと待つ）
cv2.destroyAllWindows()  # ウインドウを閉じる。