"""
Trains the specified network architecture for the given number of epochs and
desired learning rate. Ensuing, predictions are made based off test data.

Args:
1. Network architecture
	-- one of 'unet', 'alexnet', 'vgg', 'lenet', 'deepjet'
2. Batch size for training
	-- positive integer
3. Number of epochs during training
	-- positive integer
4. Learning Rate
	-- between 0 and 1
"""

import numpy as np
import sys

from keras.models import Model
from keras.layers import AveragePooling2D, BatchNormalization, Convolution2D, Dropout, Input, MaxPooling2D, merge, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import train_data, test_data
from preprocess import preprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

smooth = 1.

def dice_coef(y_true, y_pred):
	"""
	Value of the Dice Coefficient based two datasets.
	"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def train(model, batch_size, num_epochs):
	"""
	Train the selected model for a specified number of epochs and batch size.
	"""
	print '-'*30
    print 'Creating and compiling model ...'
    print '-'*30

	imgs_train, imgs_mask_train = train_data()
	imgs_train = prepreprocess(imgs_train)
	imgs_mask_train = prepreprocess(imgs_mask_train)

	imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.

    model_checkpoint = ModelCheckpoint('net.hdf5', monitor='loss', save_best_only=True)

    print '-'*30
    print 'Fitting model ...'
    print '-'*30

    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size,
    	nb_epoch=num_epochs, verbose=1, shuffle=True,
    	callbacks=[model_checkpoint], validation_split=0.1)

    print '-'*30
    print 'Training Complete'
    print '-'*30

def predict(model):
	"""
	Predicts masks on test images and saves them locally.
	"""

	imgs_test, imgs_id_test = load_test_data()
	imgs_test = preprocess(imgs_test)

	imgs_test = imgs_test.astype('float32')
	imgs_test -= mean
    imgs_test /= std

	model.load_weights('net.hdf5')

	print '-'*30
    print 'Predicting masks ...'
    print '-'*30

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    
    if not os.path.exists(os.path.join(PATH, 'masks')):
        os.mkdir(os.path.join(PATH, 'masks'))

    print '-'*30
    print 'Saving image masks ...'
    print '-'*30

    for i in range(len(imgs_mask_test)):

	    plt.imshow(np.asarray(imgs_mask_test[i][0]))
	    plt.savefig(os.path.join(PATH, 'output_mask_%d.png' % i))

if __name__ == '__main__':
	assert len(sys.argv) == 5

	architecture = eval(sys.argv[1])
	batch_size = eval(sys.argv[2])
	num_epochs = eval(sys.argv[3])
	learning_rate = eval(sys.argv[4])

	if architecture == 'unet':
		from networks import unet
		model = unet(learning_rate)
		image_rows, image_cols = 
	elif architecture == 'alexnet'
		from networks import alexnet
		model = alexnet(learning_rate)
		image_rows, image_cols = 64, 96
	elif architecture == 'vgg'
		from networks import vgg
		model = vgg(learning_rate)
	elif architecture == 'lenet5'
		from networks import lenet5
		model = lenet5(learning_rate)
	elif architecture == 'deepjet'
		from networks import deepjet
		model = deepjet(learning_rate)
	else:
		print 'Invalid network architecture specified'
		sys.exit(1)

	train(model, batch_size, num_epochs)
	predict(model)