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
	batch_size = 128 * num_workers
	num_epochs = 20
	buffer_size = 10000
	
	# Load and pre-process the mnist data
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	steps_per_epoch = int(np.ceil(len(train_images) / float(batch_size)))

	# Convert to float and normalise it, followed by reshape it
	train_images = train_images.astype(np.float32) / 255
	train_images = np.expand_dims(train_images, -1)
	test_images = test_images.astype(np.float32) / 255
	test_images = np.expand_dims(test_images, -1)

	# One-hot encoding of the labels
	train_labels = tf.one_hot(train_labels, 10)
	test_labels = tf.one_hot(test_labels, 10)

	# Create the dataset and its associated one-shot iterator.
	dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
	dataset = dataset.repeat()
	dataset = dataset.shuffle(buffer_size)
	dataset = dataset.batch(128)
	train_iterator = dataset.make_one_shot_iterator()

	# Build and compile the LeNet model with MultiWorkerMirroredStrategy 
	# to run it in distributed synchronized way
	multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	with multiworker_strategy.scope():
		lenet_model = build_and_compile_lenet_model()

	# Train the model on training set
	lenet_model.fit(train_iterator, epochs=num_epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
	# Test the model on testing set
	_, accuracy = lenet_model.evaluate(x=test_images, y=test_labels, batch_size=30)
	print('Accuracy:', accuracy)


if __name__ == "__main__":
	main()
