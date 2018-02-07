# Simple Convolutional Neural Network classifier for dogs and cats images


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

#Define batch size
batch_size = 32


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('~/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('~/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
#Print class indices
print('Class indices: ', training_set.class_indices);

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000 // batch_size,
                         epochs = 25,
                         validation_data = validation_set,
                         validation_steps = 2000 // batch_size)

#Save trained weights
classifier.save_weights('~/Convolutional_Neural_Networks/weights/model_weights.h5', overwrite = True)
