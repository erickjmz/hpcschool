#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:47:30 2017

@author: fabian
"""

#import libraries
#import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

print("Step 1\n\n")
#-------------------------------- just to see the matrixes dimensions ----------
#from keras.datasets import cifar10
#(x_train22, y_train22), (x_test22, y_test22) = cifar10.load_data()
#-------------------------------------------------------------------------------


#To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)
#directories
#root_dir = os.path.abspath('./')
#images_dir = os.path.join(root_dir, 'boneagedb')
#os.path.exists(images_dir)

#------------------------------------------------
#------------------------------------------------
#------------------------------------------------
#-----STEP 1: DATA LOADING AND PREPROCESSING

ground_truth = pd.read_csv('./train_r.csv')
image_size = 32

print("Step 2\n\n")
X = []
cont = 0
for img_name in ground_truth.id:
    image_path = ""
    try:
        image_path = './boneagedb_r/' + str(img_name) + '.png'
        
        #read rgb information
        img = imread(image_path, flatten = False, mode='RGB')
        
        #read only gray scale
        #img = imread(image_path, flatten = True)
        
        ##resizing
        if(cont == 5000):
            raise Exception('I know Python!') 
        img = imresize(img, (image_size,image_size))
        img = img.astype('float32')
        temp.append(img)
       
        cont += 1
    except:
        cont += 1
        print(image_path)
        #df[df.name != 'Tina']
        ground_truth = ground_truth[ground_truth.id != img_name]
        pass
    
print("Step 3\n\n")
X = np.stack(temp)
y = ground_truth.iloc[:, 1].values
y = np.array(y, dtype=np.float32)

#split our dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

print("Step 4\n\n")
#I need to set a tensor of 4... because of the input of Conv2D
#X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 1)
#y_train = y_train.reshape(y_train.shape[0], 1)
#X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 1)
#y_test = y_test.reshape(y_test.shape[0], 1)


#------------------------------------------------
#------------------------------------------------
#------------------------------------------------
#-----STEP 2: MODEL BUILDING
def modelBuilding():    
    #Initialising the CNN
    classifier = Sequential()
    #Step 1 - Convolution
    classifier.add(Convolution2D(filters=32, kernel_size=[3,3], input_shape=(image_size, image_size, 3), activation='relu'))
    #Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    #Step 3 - Flattening
    classifier.add(Flatten())
    #Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='linear'))
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return classifier

#------------------------------------------------
#-----STEP 3: TRAINING OUR MODEL
"""
#TODO implement this code to data augmentation!!!!!!!!!!!!!!!!!
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

classifier = modelBuilding()
# fits the model on batches with real-time data augmentation:
classifier.fit_generator(
                        train_datagen.flow(X_train, y_train, batch_size=1),
                        steps_per_epoch = int(X_train.shape[0]), 
                        epochs = 50)
"""

print("Step 5\n\n")
classifier = modelBuilding()
history = classifier.fit(X_train, y_train, batch_size = 1, epochs = 2, verbose = 1, validation_data=(X_test, y_test))


print("Step 6\n\n")
cv_size = 2
predictions_valid = classifier.predict(X_test, batch_size=50, verbose=1)
compare = pd.DataFrame(data={'original':y_test.reshape((cv_size,)),
             'prediction':predictions_valid.reshape((cv_size,))})
compare.to_csv('compare.csv')


compare = pd.read_csv('./compare.csv')
from sklearn.metrics import mean_absolute_error
y_true = compare.iloc[:,1]
y_pred = compare.iloc[:,2]
error = mean_absolute_error(y_true, y_pred)
print(error)

compare = pd.read_csv('./compare.csv')
from sklearn.metrics import mean_squared_error
y_true = compare.iloc[:,1]
y_pred = compare.iloc[:,2]
error = mean_squared_error(y_true, y_pred)
print(error)
