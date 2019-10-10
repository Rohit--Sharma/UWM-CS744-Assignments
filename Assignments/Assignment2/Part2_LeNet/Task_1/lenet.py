import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("num_workers", 1, "Number of workers.")
tf.app.flags.DEFINE_integer("task_id", 0, "Task ID of current worker")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size that will be used = this value * num_workers")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of gradient descent epochs")
FLAGS = tf.app.flags.FLAGS

# Configuration of the cluster
num_workers = FLAGS.num_workers
curr_task_idx = FLAGS.task_id

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

	model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

	return model


def main():
	batch_size = FLAGS.batch_size * num_workers
	num_epochs = FLAGS.num_epochs
	buffer_size = 10000
	
	# Load and pre-process the mnist data
	# Scaling MNIST data from (0, 255] to (0., 1.]
	def scale(image, label):
		image = tf.cast(image, tf.float32)
		image /= 255
		label = tf.one_hot(label, 10)
		return image, label

	datasets, _ = tfds.load(name='mnist',
							with_info=True,
							as_supervised=True)

	train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(buffer_size).repeat()
	train_datasets = train_datasets_unbatched.batch(batch_size)

	# Build and compile the LeNet model with MultiWorkerMirroredStrategy 
	# to run it in distributed synchronized way
	multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	with multiworker_strategy.scope():
		lenet_model = build_and_compile_lenet_model()

	# Train the model on training set
	steps_per_epoch = int(np.ceil(60000 / float(batch_size)))
	lenet_model.fit(x=train_datasets, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

	test_datasets = datasets['test'].map(scale).cache().repeat().batch(batch_size)
	options = tf.data.Options()
	options.experimental_distribute.auto_shard = False
	test_datasets = test_datasets.with_options(options)
	# Test the model on testing set
	_, accuracy = lenet_model.evaluate(x=test_datasets, steps=25)
	print('Accuracy:', accuracy)


if __name__ == "__main__":
	time_begin = datetime.now()
	main()
	time_end = datetime.now()

	training_time = time_end - time_begin
	print('Total time taken:', training_time, 's')
