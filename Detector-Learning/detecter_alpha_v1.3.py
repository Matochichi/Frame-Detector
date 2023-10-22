import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split


raw_folder_path = "resource/raw"
mask_folder_path = "resource/mask"


def load_and_preprocess_image(image_path, mask_path):
    
    raw_image = np.array(Image.open(image_path))
    mask_image = np.array(Image.open(mask_path).convert('L'))
    
    raw_image = raw_image / 255.0
    mask_image = mask_image / 255.0
    
    return raw_image, mask_image


def create_dataset(raw_folder_path, mask_folder_path):
    raw_images = []
    mask_images = []
    for filename in os.listdir(raw_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            raw_image_path = os.path.join(raw_folder_path, filename)
            mask_image_path = os.path.join(mask_folder_path, filename)
            
            if os.path.exists(mask_image_path):
                raw_image, mask_image = load_and_preprocess_image(raw_image_path, mask_image_path)
                raw_images.append(raw_image)
                mask_images.append(mask_image)
                
    return np.array(raw_images), np.array(mask_images)


def unet_model(input_size=(1000, 1000, 3)):
    inputs = keras.Input(input_size)
    
    
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    
    c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1], axis=3)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def main():
    
    raw_images, mask_images = create_dataset(raw_folder_path, mask_folder_path)
    
    
    raw_images = raw_images.reshape((-1, 1000, 1000, 3))
    mask_images = mask_images.reshape((-1, 1000, 1000, 1))
    
    
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    
    X_train, X_val, y_train, y_val = train_test_split(raw_images, mask_images, test_size=0.1, random_state=42)
    
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=1)
    
     
    model_save_dir = "models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    
    model.save(os.path.join(model_save_dir, "detector_model.h5"))

    print("Model saved successfully!")
    
    
    print("Model Evaluation:")
    model.evaluate(X_val, y_val)
    
if __name__ == "__main__":
    main()
    
    import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_new_image(image_path, target_size=(1000, 1000)):
    # 画像を読み込む
    image = Image.open(image_path)
    
    # 画像をリサイズ
    image = image.resize(target_size)
    
    # numpy配列に変換し、ピクセル値を[0, 1]の範囲に正規化
    image_array = np.array(image) / 255.0
    
    # データの形状を変更
    image_array = image_array.reshape((1,) + image_array.shape)
    
    return image_array, image

def find_frame_and_fill(model, image_path):
    # 新しい画像を読み込み、前処理
    image_array, original_image = load_and_preprocess_new_image(image_path)
    
    # モデルを使用してセグメンテーション
    prediction = model.predict(image_array)
    
    # セグメンテーション結果を二値化
    frame = prediction[0, :, :, 0] > 0.5
    
    # 元の画像をnumpy配列に変換
    original_image_array = np.array(original_image)
    
    # フレーム内側を緑色で塗りつぶす
    original_image_array[frame] = [0, 255, 0]
    
    # 結果を表示
    plt.imshow(original_image_array)
    plt.axis('off')
    plt.show()

# モデルの読み込み（モデルのパスを指定してください）
model = tf.keras.models.load_model('models/detector_model.h5')

# 新しい画像を使用してフレームの内側を見つけ、塗りつぶす
find_frame_and_fill(model, 'test.jpg')
