import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = keras.Sequential()

# Step 1 - Convolution
#input_shape goes reverse if it is theano backend
#Images are 2D
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#Most of the time it's (2,2) not loosing many. 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#Inputs are the pooled feature maps of the previous layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#relu - rectifier activation function
#128 nodes in the hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
#Sigmoid is used because this is a binary classification. For multiclass softmax
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#adam is for stochastic gradient descent 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
#Preprocess the images to reduce overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test0',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 8000, #number of images in the training set
epochs = 1,
validation_data = test_set,
validation_steps = 2000)

classifier.save('classifier.h5')

#Prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('whatami.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'

print(prediction)
