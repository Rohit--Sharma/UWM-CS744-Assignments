import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



model = models.Sequential()


model.add(layers.Conv2D(
	filters=6, 
	kernel_size=(5,5),
	stride=1,
	padding="valid",
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
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
#x = tf.placeholder(tf.float32, shape=[None, 3])
#linear_model = tf.layers.Dense(units=1)
#y = linear_model(x)

