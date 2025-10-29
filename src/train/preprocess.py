import numpy as np
import tensorflow as tf
import cv2, os
from .hyperparameters import image_path, mask_path, image_size, exclude, autotune, batch_size

# Read dataset
def read_image(image):
    path = image.numpy().decode('utf-8')
    if path not in exclude:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, 2)
        img = tf.image.resize(img, [image_size, image_size])
    else:
        img = np.zeros((image_size, image_size, 1), dtype=np.float32)
    return img

def read_mask(image):
    path = image.numpy().decode('utf-8')
    if path not in exclude:
        img = cv2.imread(path, 0)
        img = img * 257.
        img = np.expand_dims(img, 2)
        img = tf.image.resize(img, [image_size, image_size])
    else:
        img = np.zeros((image_size, image_size, 1), dtype=np.float32)
    return img

def read_image_tf(x):
    img = tf.py_function(read_image, [x], [tf.float32])[0]
    img.set_shape((image_size, image_size, 1))
    return img

def read_mask_tf(x):
    mask = tf.py_function(read_mask, [x], [tf.float32])[0]
    mask.set_shape((image_size, image_size, 1))
    return mask

dataset_image = (
    tf.data.Dataset.list_files(os.path.join(image_path, '*.tif'), shuffle=False)
    .map(read_image_tf, num_parallel_calls=autotune)
)

dataset_mask = (
    tf.data.Dataset.list_files(os.path.join(mask_path, '*.tif'), shuffle=False)
    .map(read_mask_tf, num_parallel_calls=autotune)
)

# Augmentation làm giàu data
def aug(image, label):
    seed = np.random.randint(0, 64)
    image = tf.image.random_flip_left_right(image, seed)
    image = tf.image.random_flip_up_down(image, seed)
    image = tf.image.random_contrast(image, .3, .7, seed)
    label = tf.image.random_flip_left_right(label, seed)
    label = tf.image.random_flip_up_down(label, seed)
    label = tf.image.random_contrast(label, .3, .7, seed)
    return image, label

def get_data():
    dataset = tf.data.Dataset.zip((dataset_image, dataset_mask))
    dataset = dataset.map(aug, num_parallel_calls=autotune)
    dataset = dataset.shuffle(128).batch(batch_size=batch_size).cache().prefetch(autotune)

    return dataset