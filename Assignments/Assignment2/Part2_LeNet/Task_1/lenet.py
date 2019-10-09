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
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["node0:2222", "node1:2222", "node2:2222"]
    },
   "task": {"type": "worker", "index": int(sys.argv[1])}
})


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

	# Convert to float and normalise it.
	train_images = train_images.astype(np.float32) / 255
	test_images = test_images.astype(np.float32) / 255

	# Reshape the training and test set
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	# One-hot encoding of the labels
	train_labels = to_categorical(train_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	# Build and compile the LeNet model with MultiWorkerMirroredStrategy 
	# to run it in distributed synchronized way
	multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	with multiworker_strategy.scope():
		lenet_model = build_and_compile_lenet_model()

	# Train the model on training set
	lenet_model.fit(train_images, train_labels, epochs=10, batch_size=100, validation_data=(test_images, test_labels), steps_per_epoch=2)
	# Test the model on testing set
	_, accuracy = lenet_model.evaluate(x=test_images, y=test_labels, batch_size=100)
	print('Accuracy:', accuracy)


if __name__ == "__main__":
	main()
