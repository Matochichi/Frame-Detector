import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import flatten
import matplotlib.pyplot as plt

# GPU設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ハイパーパラメータ
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4

def load_images_from_folder(folder_path, target_size=IMG_SIZE, binarize_marked=False, show_mask=False):
    images = []
    masks = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=target_size, color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        if binarize_marked:
            mask = np.where(img_array > 128/255, 1.0, 0.0)
            masks.append(mask[..., 0])
            # if show_mask:
            #     plt.imshow(mask[..., 0], cmap='gray')
            #     plt.title(filename)
            #     plt.show()
        images.append(img_array)
    if binarize_marked: 
        return np.array(images), np.array(masks)
    else:
        return np.array(images)

# def load_images_from_folder(folder_path, target_size=IMG_SIZE, binarize_marked=False):
#     images = []
#     for filename in sorted(os.listdir(folder_path)):
#         img_path = os.path.join(folder_path, filename)
#         img = load_img(img_path, target_size=target_size, color_mode="rgb")
#         img_array = img_to_array(img) / 255.0
#         if binarize_marked:
#             img_array = np.where(img_array > 128/255, 1.0, 0.0)
#         images.append(img_array)
#     return np.array(images)

raw_images = load_images_from_folder("resource\\raw_r")
marked_images, binary_masks = load_images_from_folder("resource\\mask_r", binarize_marked=True, show_mask=True)

binary_mask = marked_images[..., 0]

# U-Netモデルの定義
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    e1, p1 = encoder_block(inputs, 32)
    e2, p2 = encoder_block(p1, 64)
    e3, p3 = encoder_block(p2, 128)
    e4, p4 = encoder_block(p3, 256)
    
    b = conv_block(p4, 512)
    
    d4 = decoder_block(b, e4, 256)
    d3 = decoder_block(d4, e3, 128)
    d2 = decoder_block(d3, e2, 64)
    d1 = decoder_block(d2, e1, 32)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# データ拡張
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Dice損失とIoUメトリクスの定義
def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
    sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
    jac = (intersection + 1e-5) / (sum_ - intersection + 1e-5)
    return jac

# モデルのコンパイル
model = unet_model((IMG_SIZE[0], IMG_SIZE[1], 3))
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=dice_loss, metrics=[iou, 'binary_accuracy'])

# カスタムコールバック
model_checkpoint = ModelCheckpoint("unet_model.h5", save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir='logs')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# モデルの訓練
history = model.fit(datagen.flow(raw_images, binary_masks, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint, tensorboard, reduce_lr, early_stop])

# モデルの保存
model.save("unet_model_final.h5")

print("Training completed and model saved")
