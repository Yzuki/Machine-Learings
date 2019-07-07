import numpy as np
import math
import cv2
import random
imgsize = 96

with open('train.txt') as f1:
	train = f1.readlines()
with open('val.txt') as f2:
	test = f2.readlines()

with open('train_sbd.txt') as f3:
	train_sbd = f3.readlines()
with open('val_sbd.txt') as f4:
	test_sbd = f4.readlines()

for i in train_sbd:
	if i not in train:
		train.append(i)

for j in test_sbd:
	if j not in test:
		test.append(j)
random.shuffle(test)
print(len(train))
print(len(test)) 

train_xs = []
train_ys = []
for train_name in train:
	train_name = train_name.strip()
	train_x = cv2.resize(cv2.imread('JPEGImages/'+train_name+'.jpg'),(imgsize,imgsize))
	train_xs.append(train_x)
	y = cv2.resize(cv2.imread('SegmentationClass_aug/'+train_name+'.png'),(imgsize,imgsize),interpolation = cv2.INTER_NEAREST)
	train_y = np.zeros((imgsize,imgsize,21))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			lb = y[i,j,0]
			train_y[i,j,lb] = 1
	train_ys.append(train_y)
train_xs = np.array(train_xs)
train_ys = np.array(train_ys)
np.save('train_x.npy',train_xs)
np.save('train_y.npy',train_ys)

test_xs = []
test_ys = []
for test_name in test:
	test_name = test_name.strip()
	test_x = cv2.resize(cv2.imread('JPEGImages/'+test_name+'.jpg'),(imgsize,imgsize))
	test_xs.append(test_x)
	y = cv2.resize(cv2.imread('SegmentationClass_aug/'+test_name+'.png'),(imgsize,imgsize),interpolation = cv2.INTER_NEAREST)
	test_y = np.zeros((imgsize,imgsize,21))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			lb = y[i,j,0]
			test_y[i,j,lb] = 1
	test_ys.append(test_y)
test_xs = np.array(test_xs)
test_ys = np.array(test_ys)
np.save('test_x.npy',test_xs)
np.save('test_y.npy',test_ys)