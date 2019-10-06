import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import np_utils

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert to float and normalise it.
train_images = train_images.astype(np.float32)/255
test_images = test_images.astype(np.float32)/255

#Reshape the training and test set
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# One-hot encoding of the labels
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

model = models.Sequential()


model.add(layers.Conv2D(
	filters=6, 
	kernel_size=(5,5),
	stride=1,
	padding="same",
	activation="tanh",
	input_shape = (32, 32, 1)
	))

model.add(layes.AveragePooling2D(
	pool_size=(2,2),
	padding=0,
	stride=2,
	))

model.add(layers.Conv2D(
	filters=16,
	kernel_size=(5, 5),
	stride=1,
	padding="valid",
	activation="tanh",
	input_shape = (14, 14, 6)
	))

model.add(layers.AveragePooling2D(
	pool_size = (5,5),
	stride=2
	))

model.add(layers.Flatten())

model.add(layers.Dense(
	units=120,
	kernel_size=(1,1),
	activation="tanh"
	))

model.add(layers.Dense(
	units=84,
	activation="tanh"
	))

model.add(layers.Dense(
	units=10,
	activation="softmax"
	))


model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.fit(train_images, train_labels, epochs=10, batch_size=100, validation_data=(test_images, test_labels))
