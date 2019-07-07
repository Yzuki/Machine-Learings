import numpy as np
import math
import cv2
import random
import os
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential,Model
from keras.layers import Input, Activation, Add,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AvgPool2D, BatchNormalization,ZeroPadding2D
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
train_x = []
for i in range(len(os.listdir('train'))):
	img = cv2.imread('train/'+ str(i)+'.png',0)
	train_x.append(img)
train_x = np.array(train_x).reshape(-1,28,28,1)/255.0
test_x = []
for i in range(len(os.listdir('test'))):
	img = cv2.imread('test/'+ str(i)+'.png',0)
	test_x.append(img)
test_x = np.array(test_x).reshape(-1,28,28,1)/255.0


ftrain = open('label_train.txt')
train_y = []
for i in ftrain.readlines():
	i = i.strip()
	train_y.append(int(i))
train_y = to_categorical(np.array(train_y).reshape(-1,1))
ftrain.close()

ftest = open('label_test.txt')
test_y = []
for i in ftest.readlines():
	i = i.strip()
	test_y.append(int(i))
test_y = to_categorical(np.array(test_y).reshape(-1,1))
ftest.close()

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

def alexnet(weights_path = None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(28,28,1)))  
	model.add(Conv2D(24,(11,11),strides=(4,4),padding='valid',activation='relu',kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
	model.add(ZeroPadding2D((1,1))) 
	model.add(Conv2D(64,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(96,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(96,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(64,activation='relu'))
	model.add(Dense(10,activation='softmax'))
	model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])
	if weights_path:         
		model.load_weights(weights_path)     
	return model


def VGG(weights_path=None):       
	model = Sequential()

	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
					activation ='relu', input_shape = (28,28,1)))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
					activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))


	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))
	model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])

	if weights_path:         
		model.load_weights(weights_path)     
	return model


def identity_block(X, f, filters, stage, block):     
	conv_name_base = 'res' + str(stage) + block + '_branch'    
	bn_name_base = 'bn' + str(stage) + block + '_branch'     
	F1, F2, F3 = filters     
	X_shortcut = X     
	X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)   
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)    
	X = Activation('relu')(X)     
	X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)    
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)    
	X = Activation('relu')(X)     
	X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)    
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)    
	X = Add()([X, X_shortcut])    
	X = Activation('relu')(X)     
	return X

def convolution_block(X, f, filters, stage, block, s=2):     
	conv_name_base = 'res' + str(stage) + block + '_branch'    
	bn_name_base = 'bn' + str(stage) + block + '_branch'    
	F1, F2, F3 = filters     
	X_shortcut = X    
	X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)   
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)   
	X = Activation('relu')(X)     
	X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)    
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)    
	X = Activation('relu')(X)     
	X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)   
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)     
	X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)   
	X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut)     
	X = Add()([X, X_shortcut])   
	X = Activation('relu')(X)     
	return X

def ResNet(input_shape = (28, 28, 1), classes = 10):     
	X_input = Input(input_shape)     
	X = ZeroPadding2D((3, 3))(X_input)     
	X = Conv2D(16, (7, 7), strides = (2,2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)    
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)    
	X = Activation('relu')(X)    
	X = MaxPool2D((3, 3), strides = (2,2))(X)     
	X = convolution_block(X, f = 3, filters = [16,16,64], stage = 2, block = 'a', s = 1)    
	X = identity_block(X, 3, [16,16,64], stage=2, block='b')    
	X = identity_block(X, 3, [16,16,64], stage=2, block='c')    
	X = convolution_block(X, f = 3, filters = [32,32,128], stage = 3, block = 'a', s = 2)    
	X = identity_block(X, 3, [32,32,128], stage=3, block='b')    
	X = identity_block(X, 3, [32,32,128], stage=3, block='c')       
	X = convolution_block(X, f = 3, filters = [64,64,256], stage = 4, block = 'a', s = 2)    
	X = identity_block(X, 3, [64,64,256], stage=4, block='b')    
	X = identity_block(X, 3, [64,64,256], stage=4, block='c')            
	X = AvgPool2D((2, 2), name='avg_pool')(X)     
	X = Flatten()(X)    
	X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)     
	model = Model(inputs = X_input, outputs = X, name = 'ResNet')     
	model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
	return model


print(train_x.shape)
#m = alexnet()
m = VGG()
#m = ResNet()
print(m.summary())
from keras import callbacks
checkpoint = callbacks.ModelCheckpoint('cnn2.hdf5', monitor='val_acc', save_best_only=True)
early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=30)
annealer = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
m.fit(train_x,train_y,shuffle = True,validation_data = (test_x,test_y), batch_size=512, epochs=1000,callbacks = [annealer,checkpoint,early_stopping])
from keras.models import load_model
model = load_model("cnn2.hdf5")
final_loss, final_acc = model.evaluate(test_x,test_y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))