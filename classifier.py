from keras.models import Model
from keras.layers import Input, merge, Multiply
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from itertools import repeat


# input dimensions  
rows, cols , channels = 224,224,3    

#model params
epochs = 200
batch_size =  32
nb_classes = 2
lr = 0.002

## baseline model
def CNN_classifier(rows,cols,channels,nb_classes,lr):
	
	# conv1
	conv2d_1 = Conv2D(16, (2, 2), strides=(1, 1), activation='relu', padding='same')(input = Input(shape=(rows,cols,channels)))
	conv2d_1 = BatchNormalization(axis=-1)(conv2d_1)
	conv2d_1 = Dropout(0.30)(conv2d_1)

	#maxpool1
	MP_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_1)

	#conv2
	conv2d_2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu', padding='same')(MP_1)
	conv2d_2 = BatchNormalization(axis=-1)(conv2d_2)
	conv2d_2 = Dropout(0.30)(conv2d_2)

	#maxpool2
	MP_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_2)

	#conv3
	conv2d_3 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same')(MP_2)
	conv2d_3 = BatchNormalization(axis=-1)(conv2d_3)
	conv2d_3 = Dropout(0.30)(conv2d_3)

	#maxpool3
	MP_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_3)
	
	#conv4
	conv2d_4 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same')(MP_3)
	conv2d_4 = BatchNormalization(axis=-1)(conv2d_4)
	conv2d_4 = Dropout(0.30)(conv2d_4)

	#maxpool3
	MP_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_4)

	#flattn
	flatn = Flatten()(MP_4)

	#aggregation
	dense = Dense(256, activation='sigmoid')(flatn)
	dense = Dropout(0.50)(dense)
	dense = Dense(128, activation='sigmoid')(dense)
	dense = Dropout(0.50)(dense)
	out = Dense(nb_classes, activation='softmax', name='out')(dense)
	model = Model(input, out, name='CNN_classifier')	
	opt = Adam(lr = lr)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	
	model.summary()
	return model


CNN_classifier = CNN_classifier(rows,cols,channels,nb_classes,lr)


# feature and label
X_train = np.load('../train_feature.npy')
Y_train = np.load('../train_label.npy')
X_val = np.load('../val_feature.npy')
Y_val = np.load('../val_label.npy')

## training
def model_train(model,X_train,Y_train,X_val,Y_val,epochs,batch_size,model_name,output_folder):
	filepath=os.path.join(output_folder,model_name + '-{epoch:02d}-{val_loss:.2f}.hdf5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=callbacks_list, epochs=epochs, shuffle=False,batch_size=batch_size,verbose=2)
	return hist
	
	
train_history = model_train(CNN_classifier,X_train,Y_train,X_val,Y_val,epochs,batch_size,classifier,output_folder)



