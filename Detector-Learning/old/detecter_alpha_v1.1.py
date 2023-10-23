import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# U-Netモデルの定義
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder部分
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # 中間部分
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    # Decoder部分
    u1 = UpSampling2D((2, 2))(c3)
    m1 = Concatenate()([u1, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(m1)
    u2 = UpSampling2D((2, 2))(c4)
    m2 = Concatenate()([u2, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(m2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return Model(inputs, outputs)

# データの前処理
def preprocess_image(img_path, target_size):
    img = load_img(img_path, target_size=target_size, grayscale=True)
    img = img_to_array(img) / 255.0
    return img

def binary_mask(img, threshold=0.5):
    return (img > threshold).astype(np.float32)

# フォルダから画像を読み込む
raw_images = [preprocess_image(f'resource/raw/{filename}', (1000, 1000)) for filename in os.listdir('resource/raw')]
mask_images = [binary_mask(preprocess_image(f'resource/mask/{filename}', (1000, 1000)), 128/255) for filename in os.listdir('resource/mask')]

raw_images = np.array(raw_images)
mask_images = np.array(mask_images)

# モデルのトレーニング
model = build_unet((1000, 1000, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(raw_images, mask_images, batch_size=4, epochs=20)

model.save('frame_detection_model.h5')
