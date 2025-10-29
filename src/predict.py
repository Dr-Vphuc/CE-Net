from src.model.cenet_architecture import CE_Net
from src.model.loss_function import apply_loss
import tensorflow as tf
from src.train.preprocess import read_image, read_mask
from src.train.hyperparameters import num_files, image_path, mask_path, trained_model_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

model = CE_Net()
apply_loss(model)

model.load_weights(trained_model_path)
# model.summary()

dataset_image_only = tf.data.Dataset.list_files(image_path + '\\*.tif',shuffle=False).map(lambda x: tf.py_function(read_image,[x],[tf.float32]))
dataset_mask_only = tf.data.Dataset.list_files(mask_path+  '\\*.tif',shuffle=False).map(lambda x: tf.py_function(read_mask,[x],[tf.float32]))


take = 6
num = np.random.randint(0, int(num_files - take - 1))

image = np.array(list(dataset_image_only.skip(num).take(take).as_numpy_iterator()))
truth = np.array(list(dataset_mask_only.skip(num).take(take).as_numpy_iterator()))
pred = model.predict(dataset_image_only.skip(num).batch(take).take(1))

print(pred[0].shape)
plt.rcParams['figure.figsize'] = [20,40]
index = 1
for i in range(take):
    plt.subplot(take,3,index)
    plt.title('image %s'%i)
    plt.imshow(image[i,0,:,:,0], cmap='gray')
    index += 1
    plt.subplot(take,3,index)
    plt.title('truth %s'%i)
    plt.imshow(truth[i,0,:,:,0], cmap='gray')
    index += 1
    plt.subplot(take,3,index)
    plt.title('pred %s'%i)
    plt.imshow(pred[i,:,:,0], cmap='gray')
    index += 1

plt.tight_layout()
plt.savefig('prediction_results.png')
plt.close()

