import keras, os
import tensorflow as tf

image_path = 'C:\\Users\\ADMIN\\cenet\\Dataset\\2d_images'
mask_path = 'C:\\Users\\ADMIN\\cenet\\Dataset\\2d_masks'
trained_model_path = 'F:\\Project\\FindLungsCTimg\\model_weights.h5'

exclude = [image_path+'/ID_0254_Z_0075.tif',mask_path+'/ID_0254_Z_0075.tif',
           image_path+'/ID_0052_Z_0108.tif',mask_path+'/ID_0052_Z_0108.tif',
           image_path+'/ID_0079_Z_0072.tif',mask_path+'/ID_0079_Z_0072.tif',
           image_path+'/ID_0134_Z_0137.tif',mask_path+'/ID_0134_Z_0137.tif']

image_size = 448
num_files = len(os.listdir(image_path)) - len(exclude) / 2

lr_init = 0.0001
total_epoch = 24
keep_scale = 0.2

autotune = tf.data.experimental.AUTOTUNE

batch_size = 10
l1l2 = keras.regularizers.l1_l2(l1=0, l2=.0005)

