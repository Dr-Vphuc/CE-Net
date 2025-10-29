import keras
from keras.layers import Conv2D, BatchNormalization, Add, ReLU, MaxPool2D, Conv2DTranspose, Resizing, Concatenate
from ..train.hyperparameters import keep_scale, l1l2, image_size

# Residual block
def resblock(x,level='en_l1',filters=64,keep_scale=keep_scale,l1l2=l1l2,downsample=False,bn_act=True,first_layer=False):
    if downsample:
        if not first_layer:
            x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
            x = Conv2D(int(filters/2),(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
        else:
            x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
            x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
    else:
        x_H = x
        x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
    x = BatchNormalization(name=level+'_conv1bn')(x)
    x = ReLU(name=level+'_conv1relu')(x)
    if downsample:
        x = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
    else:
        x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
    x = BatchNormalization(name=level+'_conv2bn')(x)
    x = ReLU(name=level+'_conv2relu')(x)
    x = Add(name=level+'_add')([x_H*keep_scale,x*(1-keep_scale)])
    if bn_act:
        x = BatchNormalization(name=level+'_finalbn')(x)
        x = ReLU(name=level+'_finalrelu')(x)
    return x
###

def DAC_block(layer):
    b1 = Conv2D(512, (3, 3), padding='same', dilation_rate=1, name='dac_b1_conv1', activation=None)(layer)

    b2 = Conv2D(512, (3, 3), padding='same', dilation_rate=3, name='dac_b2_conv1', activation=None)(layer)
    b2 = Conv2D(512, (1, 1), padding='same', dilation_rate=1, name='dac_b2_conv2', activation=None)(b2)

    b3 = Conv2D(512, (3, 3), padding='same', dilation_rate=1, name='dac_b3_conv1', activation=None)(layer)
    b3 = Conv2D(512, (3, 3), padding='same', dilation_rate=3, name='dac_b3_conv2', activation=None)(b3)
    b3 = Conv2D(512, (1, 1), padding='same', dilation_rate=1, name='dac_b3_conv3', activation=None)(b3)

    b4 = Conv2D(512, (3, 3), padding='same', dilation_rate=1, name='dac_b4_conv1', activation=None)(layer)
    b4 = Conv2D(512, (3, 3), padding='same', dilation_rate=3, name='dac_b4_conv2', activation=None)(b4)
    b4 = Conv2D(512, (3, 3), padding='same', dilation_rate=5, name='dac_b4_conv3', activation=None)(b4)
    b4 = Conv2D(512, (1, 1), padding='same', dilation_rate=1, name='dac_b4_conv4', activation=None)(b4)

    layer = Add(name='dac_add')([layer, b1, b2, b3, b4])
    layer = ReLU(name='dac_relu')(layer)

    return layer

def RMP_block(layer):
    b1 = MaxPool2D((2, 2), strides=(2, 2), padding='valid', name='rmp_b1_pool')(layer)
    b1 = Conv2D(1, (1, 1), padding='valid', name='rmb_b1_conv1', activation=None)(b1)
    b1 = Conv2DTranspose(1, (1, 1), (2, 2), padding='valid', kernel_regularizer=l1l2, activation=None)(b1)
    b1 = Resizing(14, 14, interpolation='bilinear', name='rmp_b1_resize')(b1)

    b2 = MaxPool2D((3, 3), strides=(3, 3), padding='valid', name='rmp_b2_pool')(layer)
    b2 = Conv2D(1, (1, 1), padding='valid', name='rmb_b2_conv1', activation=None)(b2)
    b2 = Conv2DTranspose(1, (1, 1), (3, 3), padding='valid', kernel_regularizer=l1l2, activation=None)(b2)
    b2 = Resizing(14, 14, interpolation='bilinear', name='rmp_b2_resize')(b2)

    b3 = MaxPool2D((5, 5), strides=(5, 5), padding='valid', name='rmp_b3_pool')(layer)
    b3 = Conv2D(1, (1, 1), padding='valid', name='rmb_b3_conv1', activation=None)(b3)
    b3 = Conv2DTranspose(1, (1, 1), (5, 5), padding='valid', kernel_regularizer=l1l2, activation=None)(b3)
    b3 = Resizing(14, 14, interpolation='bilinear', name='rmp_b3_resize')(b3)

    b4 = MaxPool2D((6, 6), strides=(6, 6), padding='valid', name='rmp_b4_pool')(layer)
    b4 = Conv2D(1, (1, 1), padding='valid', name='rmb_b4_conv1', activation=None)(b4)
    b4 = Conv2DTranspose(1, (1, 1), (6, 6), padding='valid', kernel_regularizer=l1l2, activation=None)(b4)
    b4 = Resizing(14, 14, interpolation='bilinear', name='rmp_b4_resize')(b4)

    layer = Concatenate(name='rmp_concat')([layer, b1, b2, b3, b4])
    layer = ReLU(name='rmp_relu')(layer)

    return layer


def CE_Net():
    # Define CE-Net
    ##Inputs
    inputs = keras.Input(shape=(image_size, image_size, 1), name='input')

    layer_0 = Conv2D(64, (7, 7), (2, 2), padding='same', name='input_conv1', activation=None)(inputs)
    layer = BatchNormalization(name='input_conv1bn')(layer_0)
    layer = ReLU(name='input_conv1relu')(layer)

    ## Encoder block
    layer = resblock(layer, 'en_l1', 64, keep_scale, l1l2, downsample=True, bn_act=True, first_layer=True)
    layer = resblock(layer, 'en_l2', 64, keep_scale, l1l2, downsample=False, bn_act=True)
    encoder_1 = resblock(layer, 'en_l3', 64, keep_scale, l1l2, downsample=False, bn_act=False)
    layer = BatchNormalization(name='en_l3_finalbn')(encoder_1)
    layer = ReLU(name='en_l3_finalrelu')(layer)

    layer = resblock(layer, 'en_l4', 128, keep_scale, l1l2, downsample=True, bn_act=True)
    layer = resblock(layer, 'en_l5', 128, keep_scale, l1l2, downsample=False, bn_act=True)
    layer = resblock(layer, 'en_l6', 128, keep_scale, l1l2, downsample=False, bn_act=True)
    encoder_2 = resblock(layer, 'en_l7', 128, keep_scale, l1l2, downsample=False, bn_act=False)
    layer = BatchNormalization(name='en_l7_finalbn')(encoder_2)
    layer = ReLU(name='en_l7_finalrelu')(layer)

    layer = resblock(layer, 'en_l8', 256, keep_scale, l1l2, downsample=True, bn_act=True)
    layer = resblock(layer, 'en_l9', 256, keep_scale, l1l2, downsample=False, bn_act=True)
    layer = resblock(layer, 'en_l10', 256, keep_scale, l1l2, downsample=False, bn_act=True)
    layer = resblock(layer, 'en_l11', 256, keep_scale, l1l2, downsample=False, bn_act=True)
    layer = resblock(layer, 'en_l12', 256, keep_scale, l1l2, downsample=False, bn_act=True)
    encoder_3 = resblock(layer, 'en_l13', 256, keep_scale, l1l2, downsample=False, bn_act=False)
    layer = BatchNormalization(name='en_l13_finalbn')(encoder_3)
    layer = ReLU(name='en_l13_finalrelu')(layer)

    layer = resblock(layer, 'en_l14', 512, keep_scale, l1l2, downsample=True, bn_act=True)
    layer = resblock(layer, 'en_l15', 512, keep_scale, l1l2, downsample=False, bn_act=True)
    layer = resblock(layer, 'en_l16', 512, keep_scale, l1l2, downsample=False, bn_act=True)

    ## DAC block
    layer = DAC_block(layer)
    ## RMP block
    layer = RMP_block(layer)

    ## Decoder block
    layer = Conv2D(256, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l1_conv1', activation=None)(layer)
    layer = BatchNormalization(name='de_l1_conv1bn')(layer)
    layer = ReLU(name='de_l1_conv1relu')(layer)
    layer = Conv2DTranspose(256, (3, 3), (2, 2), padding='same', kernel_regularizer=l1l2, output_padding=1,
                            name='de_l1_deconv2', activation=None)(layer)
    layer = BatchNormalization(name='de_l1_conv2bn')(layer)
    layer = ReLU(name='de_l1_deconv2relu')(layer)
    layer = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l1l2, name='de_l1_conv3', activation=None)(layer)
    layer = Add(name='de_l1_add')([encoder_3 * keep_scale, layer * (1 - keep_scale)])
    layer = BatchNormalization(name='de_l1_conv3bn')(layer)
    layer = ReLU(name='de_l1_conv3relu')(layer)

    layer = Conv2D(128, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l2_conv1', activation=None)(layer)
    layer = BatchNormalization(name='de_l2_conv1bn')(layer)
    layer = ReLU(name='de_l2_conv1relu')(layer)
    layer = Conv2DTranspose(128, (3, 3), (2, 2), padding='same', kernel_regularizer=l1l2, output_padding=1,
                            name='de_l2_deconv2', activation=None)(layer)
    layer = BatchNormalization(name='de_l2_conv2bn')(layer)
    layer = ReLU(name='de_l2_deconv2relu')(layer)
    layer = Conv2D(128, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l2_conv3', activation=None)(layer)
    layer = Add(name='de_l2_add')([encoder_2 * keep_scale, layer * (1 - keep_scale)])
    layer = BatchNormalization(name='de_l2_conv3bn')(layer)
    layer = ReLU(name='de_l2_conv3relu')(layer)

    layer = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l3_conv1', activation=None)(layer)
    layer = BatchNormalization(name='de_l3_conv1bn')(layer)
    layer = ReLU(name='de_l3_conv1relu')(layer)
    layer = Conv2DTranspose(64, (3, 3), (2, 2), padding='same', kernel_regularizer=l1l2, output_padding=1,
                            name='de_l3_deconv2', activation=None)(layer)
    layer = BatchNormalization(name='de_l3_conv2bn')(layer)
    layer = ReLU(name='de_l3_deconv2relu')(layer)
    layer = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l3_conv3', activation=None)(layer)
    layer = Add(name='de_l3_add')([encoder_1 * keep_scale, layer * (1 - keep_scale)])
    layer = BatchNormalization(name='de_l3_conv3bn')(layer)
    layer = ReLU(name='de_l3_conv3relu')(layer)

    layer = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l4_conv1', activation=None)(layer)
    layer = BatchNormalization(name='de_l4_conv1bn')(layer)
    layer = ReLU(name='de_l4_conv1relu')(layer)
    layer = Conv2DTranspose(64, (3, 3), (2, 2), padding='same', kernel_regularizer=l1l2, output_padding=1,
                            name='de_l4_deconv2', activation=None)(layer)
    layer = BatchNormalization(name='de_l4_conv2bn')(layer)
    layer = ReLU(name='de_l4_deconv2relu')(layer)
    layer = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l1l2, name='de_l4_conv3', activation=None)(layer)
    layer = Add(name='de_l4_add')([layer_0 * keep_scale, layer * (1 - keep_scale)])
    layer = BatchNormalization(name='de_l4_conv3bn')(layer)
    layer = ReLU(name='de_l4_conv3relu')(layer)

    ## Outputs
    layer = Conv2DTranspose(32, (3, 3), (2, 2), padding='same', kernel_regularizer=l1l2, name='final_deconv1',
                            activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1l2, name='final_conv1', activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    outputs = Conv2D(1, (3, 3), padding='same', name='output', activation=None)(layer)

    model = keras.Model(inputs, outputs, name='CE-Net')
    # print(model.summary())

    return model
