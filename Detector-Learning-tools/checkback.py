import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    # 画像を読み込む
    img = cv2.imread(image_path)

    # 赤い線を検出するためのHSV範囲を定義
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2

    # 輪郭を見つける
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を見つける（これが赤い線の領域であると仮定）
    max_contour = max(contours, key=cv2.contourArea)

    # 画像全体を赤く塗りつぶす
    filled_red = np.ones_like(img) * [0, 0, 255]

    # 赤い線の内側を元の画像の色で塗りつぶす
    cv2.drawContours(filled_red, [max_contour], -1, (255,255,255), thickness=-1)

    # 元の画像と合成
    result = cv2.bitwise_and(filled_red, img)

    # 保存
    cv2.imwrite(output_path, result)

# フォルダ内の画像を処理
folder_path = "resource\marked"
output_folder = "resource\processed"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(folder_path):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(folder_path, file_name)
        output_path = os.path.join(output_folder, file_name)
        process_image(input_path, output_path)

print("Processing completed!")
