import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import scipy


#param
IMG_SIZE = (1000, 1000)
BATCH_SIZE = 10
EPOCHS = 100

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

def load_images_from_folder(folder_path, target_size=IMG_SIZE):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=target_size, color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

raw_images = load_images_from_folder("resourceTest\\raw")
marked_images = load_images_from_folder("resourceTest\\marked")

red_channel = marked_images[..., 0]
green_channel = marked_images[..., 1]
blue_channel = marked_images[..., 2]
binary_mask = np.where((red_channel > 0.5) & (green_channel < 0.5) & (blue_channel < 0.5), 1.0, 0.0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    Dropout(0.5),
    
    tf.keras.layers.Dense(IMG_SIZE[0]*IMG_SIZE[1], activation='sigmoid'),
    tf.keras.layers.Reshape((IMG_SIZE[0], IMG_SIZE[1]))
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.fit(datagen.flow(raw_images, binary_mask, batch_size=BATCH_SIZE),
          epochs=EPOCHS,
          callbacks=[reduce_lr, early_stop])

model.save("detector_alpha_model.h5")

print ("save completed")