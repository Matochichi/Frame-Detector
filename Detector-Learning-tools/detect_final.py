import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 画像を前処理する関数
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 画像にフレーム位置をハイライトする関数
def highlight_frame(original_img, mask, alpha=0.5, color=[0, 1, 0]): # 緑色でハイライト
    highlighted = original_img.copy()
    for c in range(3):
        highlighted[..., c] = original_img[..., c] * (1 - alpha * mask) + alpha * mask * color[c]
    return highlighted

# 予測とハイライト処理を行う関数
def predict_and_highlight(model, image_path):
    original_img = img_to_array(load_img(image_path, color_mode="rgb")) / 255.0
    processed_img = preprocess_image(image_path)
    
    prediction = model.predict(processed_img)
    mask = prediction[0, :, :, 0] > 0.5
    
    return highlight_frame(original_img, mask)

# 画像を保存する関数
def save_image(img_array, save_path):
    img_array = (img_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

# モデルの読み込み
model = tf.keras.models.load_model('frame_detection_model.h5')

# testフォルダ内の全ての画像ファイルのパスを取得
image_files = [f for f in os.listdir('test') if os.path.isfile(os.path.join('test', f))]

# 画像ファイルごとにハイライト処理を行い、結果を保存
for image_file in image_files:
    image_path = os.path.join('test', image_file)
    highlighted_image = predict_and_highlight(model, image_path)
    
    # 結果を保存するパスを設定（元のファイル名に'_highlighted'を追加）
    save_path = os.path.join('test', os.path.splitext(image_file)[0] + '_highlighted.jpg')
    save_image(highlighted_image, save_path)
