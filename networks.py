"""
Implementation of U-Net, AlexNet, VGG-16, LeNet-5 and DeepJet network
architectures for image segmentation.
"""

from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD

def unet(learning_rate):
    inputs = Input((1, 64, 80))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='relu')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=SGD(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def alexnet(learning_rate):
    inputs = Input((1, 64, 96))
    
    # NOTE: AlexNet implmentation has stride = 4
    conv1 = Convolution2D(96, 11, 11, activation='relu', border_mode='same', subsample=(1, 1))(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    norm1 = BatchNormalization(epsilon=1e-4, momentum=0.9)(pool1)
    
    conv2 = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(norm1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    norm2 = BatchNormalization(epsilon=1e-4, momentum=0.9)(pool2)

    conv3 = Convolution2D(384, 3, 3, activation='relu', border_mode='same')(norm2)
    conv4 = Convolution2D(384, 3, 3, activation='relu', border_mode='same')(conv3)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)
    drop1 = Dropout(0.5)(conv6)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(drop1)
    drop2 = Dropout(0.5)(conv7)
    
    upsample1 = merge([UpSampling2D(size=(2, 2))(drop2), conv2], mode='concat', concat_axis=1)
    upsample2 = merge([UpSampling2D(size=(2, 2))(upsample1), conv1], mode='concat', concat_axis=1)
    
    conv8 = Convolution2D(1, 1, 1, activation='relu')(upsample2)

    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=SGD(lr=learning_rate, decay=5e-4, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def vgg(learning_rate):
    inputs = Input((1, 64, 96))
    
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    conv8 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv9 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv8)
    conv10 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv12 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv11)
    conv13 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv12)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    conv14 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool5)
    conv15 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv14)
    conv16 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv15)   

    up1 = merge([UpSampling2D(size=(2, 2))(conv16), conv11], mode='concat', concat_axis=1)
    up2 = merge([UpSampling2D(size=(2, 2))(up1), conv8], mode='concat', concat_axis=1)
    up3 = merge([UpSampling2D(size=(2, 2))(up2), conv5], mode='concat', concat_axis=1)
    up4 = merge([UpSampling2D(size=(2, 2))(up3), conv3], mode='concat', concat_axis=1)
    up5 = merge([UpSampling2D(size=(2, 2))(up4), conv1], mode='concat', concat_axis=1)
    
    conv_out = Convolution2D(1, 1, 1, activation='relu', border_mode='same', subsample=(1, 1))(up5)
    
    model = Model(input=inputs, output=conv_out)
    model.compile(optimizer=SGD(lr=learning_rate, decay=5e-4, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def lenet5(learning_rate):
    inputs = Input((1, 56, 56))

    conv1 = Convolution2D(6, 5, 5, activation='tanh')(inputs)
    subsample1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(16, 5, 5, activation='tanh')(subsample1)
    subsample2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(120, 5, 5, activation='tanh')(subsample2)
    conv4 = Convolution2D(84, 5, 5, activation='tanh', border_mode='same')(conv3)

    upsample = merge([UpSampling2D(size=(8, 8))(conv4), inputs], mode='concat', concat_axis=1)
    conv_out = Convolution2D(1, 1, 1, activation='tanh', border_mode='same')(upsample)

    model = Model(input=inputs, output=conv_out)
    model.compile(optimizer=SGD(lr=learning_rate, decay=2e-4, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def deepjet(learning_rate):
    inputs = Input((1, 64, 96))
    
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

    conv4_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)
    
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_3)

    upsample = merge([UpSampling2D(size=(32, 32))(pool5), conv1_1], mode='concat', concat_axis=1)
    conv_out = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(upsample)

    model = Model(input=inputs, output=conv_out)
    model.compile(optimizer=SGD(lr=learning_rate, decay=2e-4, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    return model