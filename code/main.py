# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv2
import rad_sa_regionima as regioni
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
pylab.rcParams['figure.figsize'] = 6, 4

'''Train a simple deep NN on the MNIST dataset.
Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
#from __future__ import print_function
np.random.seed(1337)  # for reproducibility

def testBinarizujTrening(X_trening):
    for idx, elem in enumerate(X_trening):
        if idx > 10:
            return
        print idx
        for jdx, pix in enumerate(elem):
            if pix > 0.1:
                X_trening[idx,jdx] = 1
            else:
                X_trening[idx,jdx] = 0

def binarizujTrening(X_trening):
    for idx, elem in enumerate(X_trening):
        print idx
        for jdx, pix in enumerate(elem):
            if pix > 0.1:
                X_trening[idx,jdx] = 1
            else:
                X_trening[idx,jdx] = 0
    return
        

from keras.datasets import mnist
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU

batch_size = 128
nb_classes = 10
nb_epoch = 10
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


plt.figure()
plt.imshow(X_test[0], 'gray')
plt.figure()
plt.imshow(X_test[1], 'gray')
plt.figure()
plt.imshow(X_test[2], 'gray')
plt.figure()
plt.imshow(X_test[3], 'gray')
plt.figure()
plt.imshow(X_test[4], 'gray')


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#print "PRE BINARIZACIJE 1 PRIMER: "
#print X_train[0]
#binarizujTrening(X_train)
#print "POSLE BINARIZACIJE 1 PRIMER: "
print X_train[0]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

ann = Sequential()
ann.add(Dense(output_dim=800, input_dim=784))
#ann.add(PReLU(init='zero', weights=None))
ann.add(Activation('relu'))
#ann.add(Dropout(0.2))
#ann.add(Dense(output_dim=128))
#ann.add(PReLU(init='zero', weights=None))
#ann.add(Dropout(0.2))
ann.add(Dense(output_dim=10))
ann.add(Activation('softmax'))
rms = RMSprop()
#sgd = SGD()
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ann.compile(loss='categorical_crossentropy', optimizer=rms)
ann.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1,
          validation_data=(X_test, Y_test))
score = ann.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


(X_train, y_train), (X_test, y_test) = mnist.load_data()
for i in range(61, 100):
    plt.figure()
    plt.imshow(X_test[i], 'gray')
"""
cap = cv2.VideoCapture(1)
cap.release()
cap.open(0)
prolaz = 1
while(True):
    print "\n\nPROLAZ BROJ ", prolaz, "\n\n"
    # Uzima frejm po frejm
    #time.sleep(2)
    ret, frame = cap.read()
    # Operacije nad frejmom
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bin_frame = 255-cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 14)
    cv2.imshow('Black and white', bin_frame)
    #cv2.imshow('Camera frame', frame)
    #img, contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im, contours = regioni.regioni_od_interesa(frame, bin_frame, prolaz, ann)
    #cv2.drawContours(img, contours, -1, (0,0,255), 1)
    #cv2.imshow('Live HW Determinant Solver',img)
    cv2.imshow('Image', frame)
    #cv2.imshow('Grayscale', gray_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # KAD SE STISNE ESC IZLAZAK IZ APLIKACIJE
        break
    prolaz = prolaz + 1

# Kada je sve gotovo, oslobodi izvor
cap.release()
cv2.destroyAllWindows()