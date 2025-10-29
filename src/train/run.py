import keras.regularizers
import cv2, math, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.model.loss_function import apply_loss
from src.model.cenet_architecture import CE_Net
from src.train.preprocess import get_data

plt.rcParams['figure.figsize'] = [16,8]
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
from hyperparameters import  batch_size, total_epoch, lr_init, num_files

dataset = get_data()

# Split train-test set
total = int(num_files)
train_size = int(total * .8)
train = dataset.take(train_size // batch_size)
test = dataset.skip(train_size // batch_size)
###

#Define model
model = CE_Net()

# Draw model structure in PNG
# keras.utils.plot_model(model, show_shapes=True)

# 3 schedulers for training
def scheduler_1(epoch):
    epoch += 1

    if epoch <= 4:
        return lr_init
    if 5 <= epoch <= 10:
        return lr_init - lr_init * math.exp(0.25 * (epoch - 8)) / 40
    elif 11 <= epoch <= 50:
        return lr_init * math.exp(-0.05 * (epoch - 10))
    else:
        return scheduler_1(50 - 1)

def scheduler_2(epoch):
    epoch += 1

    if epoch == 1:
        return lr_init
    elif 2 <= epoch <= 35:
        return (0.25 * epoch ** 3) * math.exp(-0.25 * epoch) * lr_init
    else:
        return scheduler_2(35 - 1)

def scheduler_3(epoch):
    if epoch == 0:
        return lr_init
    else:
        return lr_init * ((1 - epoch / 100) ** 0.9)

#Apply loss function
apply_loss(model)

#Load ResNet-34 pre-trained model
model.load_weights('F:\\Project\\FindLungsCTimg\\model_weights.h5')

scheduler = keras.callbacks.LearningRateScheduler(scheduler_3, verbose=1)
earlystop = keras.callbacks.EarlyStopping(monitor='val_iou',mode='max',verbose=1,patience=15,restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('../../temp.weights.h5', save_weights_only=True, monitor='val_iou', mode='max', save_best_only=True)

history = model.fit(
    train,
    batch_size=batch_size,
    epochs=total_epoch,
    callbacks=[scheduler,earlystop,checkpoint],
    validation_data=test,
    steps_per_epoch=int(0.8*num_files//batch_size),
    validation_steps=int(0.2*num_files//batch_size),
    shuffle=True
)

model.save('model.h5')
model.save_weights('model1.weights.h5')
