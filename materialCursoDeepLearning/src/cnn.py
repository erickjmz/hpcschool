# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution, input image has 64x64 pixels
#32 filters of 3x3, by default keras does not do zero padding, changing dimensions
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#RELU activation function
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening (vectorization)
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#loss function: error function to be minimized, usually is the MSE, but the crossentropy converges faster
#for weight optimization ADAM is used, a mod. of stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Data augmentation and model training

from keras.preprocessing.image import ImageDataGenerator
# Data augmentation: performs pixel wise normalization (0-1), shear, size and flipping
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

								   #Test images are also normalized
test_datagen = ImageDataGenerator(rescale = 1./255)

#We specify the training data set path, a folder per class
#Batch size: how many samples are randomly taken before doing the back propagation (32 samples are used to calculate the error and recalculate the weights)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
												 
#We specify the test data set path, random samples are taken from this folder

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#We train the model
#steps per epoch: number of samples taken per epoch, if the number is higher than the samples available, data augmentation is performed
#epochs: one epoch consists in a complete iteration over the whole data set

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)