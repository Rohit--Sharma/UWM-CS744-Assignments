import os
import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Configuration of the cluster
num_workers = int(sys.argv[1])
curr_task_idx = int(sys.argv[2])

cluster_conf = {
	'cluster': {
		'worker': []
	},
	'task': {
		'type': 'worker',
		'index': curr_task_idx
	}
}

for worker_idx in range(num_workers):
	cluster_conf['cluster']['worker'].append('node{0}:2222'.format(worker_idx))

os.environ["TF_CONFIG"] = json.dumps(cluster_conf)


# Define the LeNet model and compile it. This has to be done in a Distributed strategy
def build_and_compile_lenet_model():
	model = Sequential()

	model.add(Conv2D(
		filters=6, 
		kernel_size=(5,5),
		strides=1,
		padding="same",
		activation="tanh",
		input_shape = (28, 28, 1)
	))

	model.add(AveragePooling2D(
		pool_size=(2,2),
		strides=2,
	))

	model.add(Conv2D(
		filters=16,
		kernel_size=(5, 5),
		strides=1,
		padding="valid",
		activation="tanh",
		input_shape = (14, 14, 6)
	))

	model.add(AveragePooling2D(
		pool_size = (5,5),
		strides=2
	))

	model.add(Flatten())

	model.add(Dense(
		units=120,
		activation="tanh"
	))

	model.add(Dense(
		units=84,
		activation="tanh"
	))

	model.add(Dense(
		units=10,
		activation="softmax"
	))

	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	return model


def main():
	# Load and pre-process the mnist data
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	print(len(test_images))
	val_images = test_images[:9000]
	val_labels = test_labels[:9000]
	print(len(val_images))

	test_images = test_images[9000:9900]
	test_labels = test_labels[9000:9900]

	# Convert to float and normalise it.
	train_images = train_images.astype(np.float32) / 255
	test_images = test_images.astype(np.float32) / 255

	# Reshape the training and test set
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	# One-hot encoding of the labels
	train_labels = to_categorical(train_labels, 10)
	val_labels = to_categorical(val_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	# Build and compile the LeNet model with MultiWorkerMirroredStrategy 
	# to run it in distributed synchronized way
	multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	with multiworker_strategy.scope():
		lenet_model = build_and_compile_lenet_model()

	# Train the model on training set
	lenet_model.fit(train_images, train_labels, epochs=3, batch_size=300, validation_data=(val_images, val_labels), steps_per_epoch=2)
	# Test the model on testing set
	_, accuracy = lenet_model.evaluate(x=test_images, y=test_labels, batch_size=30)
	print('Accuracy:', accuracy)


if __name__ == "__main__":
	main()
