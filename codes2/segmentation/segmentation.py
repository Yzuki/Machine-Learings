import numpy as np
import math
import cv2
import random
import os
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential,Model
from keras.layers import Input,Reshape, UpSampling2D,Activation, Conv2DTranspose,Add,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AvgPool2D, BatchNormalization,ZeroPadding2D
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
imgsize = 96
train_x = np.load('train_x.npy').reshape(-1,imgsize,imgsize,3)
train_y = np.load('train_y.npy').reshape(-1,imgsize*imgsize,21)
test_x = np.load('test_x.npy').reshape(-1,imgsize,imgsize,3)
test_y = np.load('test_y.npy').reshape(-1,imgsize*imgsize,21)

print(train_x.shape[0])
print(test_x.shape[0])
def VGG(weights_path=None):       
	model = Sequential()

	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
					activation ='relu', input_shape = (imgsize,imgsize,3)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
					activation ='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(filters = 256, kernel_size = (7,7),padding = 'Same', 
					activation ='relu'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 256, kernel_size = (1,1),padding = 'Same', 
					activation ='relu'))
	model.add(Dropout(0.5))
	model.add(Conv2D(filters = 21, kernel_size = (1,1),padding = 'same', 
					activation ='relu'))
	model.add(Conv2DTranspose(filters = 21, kernel_size = (8,8),strides = (8,8),padding = 'valid', 
					activation =None))
	model.add(Reshape((-1,21)))
	model.add(Activation('softmax'))
	model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])
	if weights_path:         
		model.load_weights(weights_path)     
	return model


#m = alexnet()
m = VGG()
#m = ResNet()
print(m.summary())
from keras import callbacks
checkpoint = callbacks.ModelCheckpoint('cnn.hdf5', monitor='val_acc', save_best_only=True)
early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=30)
annealer = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
m.fit(train_x,train_y,shuffle = True,validation_data = (test_x,test_y), batch_size=64, epochs=1000,callbacks = [annealer,checkpoint,early_stopping])
from keras.models import load_model
model = load_model("cnn.hdf5")
final_loss, final_acc = model.evaluate(test_x,test_y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

#visualization code
"""
goodimage = []
for k in goodimage:
	img = test_x[k].reshape((-1,imgsize,imgsize,3))
	label = test_y[k].reshape(imgsize,imgsize,21)
	pred = model.predict(img).reshape(imgsize,imgsize,21)
	img_show = np.zeros((imgsize,imgsize,3))
	lb = -1
	for i in range(imgsize):
		for j in range(imgsize):
			if np.argmax(pred[i,j]) == 0:
				img_show[i,j] = np.array([0,0,0])
			elif(np.argmax(pred[i,j]) == np.argmax(label[i,j])):
				img_show[i,j] = np.array([255,255,255])
			elif(np.argmax(pred[i,j]) != np.argmax(label[i,j]) and np.argmax(label[i,j])!=0):
				img_show[i,j] = np.array([0,0,255])
			else:
				img_show[i,j] = np.array([255,255,255])

	cv2.imwrite('res/real'+str(k)+'.jpg',img.reshape((imgsize,imgsize,3)))
	cv2.imwrite('res/segmentation'+str(k)+'.jpg',img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""