import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



model = models.Sequential()


model.add(layers.Conv2D(
	filters=6, 
	kernel_size=(5,5), 
	padding=0,
	stride=1,
	input_shape = (32, 32, 1)
	))

model.add(layes.MaxPooling2D(
	filters=6,
	kernel_size=(2,2),
	padding=0,
	stride=2,
	input_shape= (28, 28, 6)
	))

model.add(layers.Conv2D(
	filters=16,
	kernel_size=(10, 10),
	stride=1,
	padding=0
	))

model.add(layers.MaxPooling2D(
	filters=16,
	kernel_size = (5,5),
	stride=1
	))

model.add(layers.Dense(
	filters=120,
	kernel_size=(1,1)
	))

model.add(layers.Dense(
	filters=84
	))

model.add(layers.Dense(
	filters=10
	))

#x = tf.placeholder(tf.float32, shape=[None, 3])
#linear_model = tf.layers.Dense(units=1)
#y = linear_model(x)

