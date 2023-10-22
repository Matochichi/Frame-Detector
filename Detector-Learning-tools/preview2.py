import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 新しい画像を読み込む
new_image = cv2.imread('test.jpg')
original_image = new_image.copy()

# 画像をモデルの入力サイズにリサイズ
new_image = cv2.resize(new_image, (1000, 1000))

# 画像のピクセル値を0-1の範囲にスケーリング
new_image = new_image / 255.0

# 画像の次元を拡張 (1, height, width, channels)
new_image = np.expand_dims(new_image, axis=0)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)

# 学習済みモデルを読み込む
model = load_model('frame_filling_model.h5', custom_objects={'dice_loss': dice_loss})

# 画像に対して予測を行う
predictions = model.predict(new_image)

# 予測結果を元のサイズにリサイズ
predicted_mask = cv2.resize(predictions[0], (original_image.shape[1], original_image.shape[0]))

# 二値化
_, predicted_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)

# 輪郭を見つける
contours, _ = cv2.findContours((predicted_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 元の画像に輪郭を赤色で描画
for contour in contours:
    cv2.drawContours(original_image, [contour], -1, (0, 0, 255), 2)

# 結果を表示
cv2.imshow('Result', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
