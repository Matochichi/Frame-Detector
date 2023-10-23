import tensorflow as tf
import cv2
import os
import numpy as np

# Paths to folders
raw_folder = "resource/raw/"
marked_folder = "resource/marked/"

# Load images
raw_images = []
marked_images = []

for filename in os.listdir(raw_folder):
    raw_img = cv2.imread(os.path.join(raw_folder, filename))
    marked_img = cv2.imread(os.path.join(marked_folder, filename))
    
    raw_images.append(raw_img)
    marked_images.append(marked_img)

raw_images = tf.convert_to_tensor(raw_images, dtype=tf.float32) / 255.0
marked_images = tf.convert_to_tensor(marked_images, dtype=tf.float32) / 255.0

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Expansive path
    u2 = UpSampling2D((2, 2))(c3)
    merge2 = Concatenate()([u2, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)

    u3 = UpSampling2D((2, 2))(c4)
    merge3 = Concatenate()([u3, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge3)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

model = unet_model((256, 256, 3))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(raw_images, marked_images, epochs=10, batch_size=8)

model.save("frame_filling_model.h5")

# Load the model
loaded_model = tf.keras.models.load_model("frame_filling_model.h5")

# Load a new image
new_image_path = "test.jpg"
new_image = cv2.imread(new_image_path)
new_image_resized = cv2.resize(new_image, (256, 256))
new_image_normalized = new_image_resized / 255.0

# Predict using the model
prediction = loaded_model.predict(new_image_normalized[np.newaxis, ...])
output_image = (prediction[0] * 255).astype(np.uint8)

# Save or display the output image
cv2.imwrite("output_image.jpg", output_image)

