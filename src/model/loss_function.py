import keras
import keras.backend as K
import tensorflow as tf
from src.train.hyperparameters import lr_init

# Loss function define
smooth = K.epsilon()
threshold = 0.8
label_smoothing = 0.0
bce_weight = 0.5

def dice_coeff(y_true, y_pred):
    numerator = tf.math.reduce_sum(y_true * y_pred) + smooth
    denominator = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred) + smooth
    return tf.math.reduce_mean(2.* numerator / denominator) * 448 * 448

def dice_loss(y_true, y_pred):
    return - dice_coeff(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return bce_weight * keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=label_smoothing) - dice_coeff(y_true, y_pred)

def bce_dice_loss_try(y_true, y_pred):
    x = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=label_smoothing)
    y = -dice_coeff(y_true, y_pred)
    return x + x/(x+y)*y

def iou(y_true, y_pred):
    y_true = tf.cast(y_true > threshold, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

loss = bce_dice_loss
optimizer = keras.optimizers.Adam(lr_init)
metrics = [keras.metrics.BinaryCrossentropy(from_logits=False,label_smoothing=label_smoothing,dtype=tf.float32,name='bce'), dice_coeff, iou]

def apply_loss(model):
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
