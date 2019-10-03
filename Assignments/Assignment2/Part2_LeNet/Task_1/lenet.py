import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#x = tf.placeholder(tf.float32, shape=[None, 3])
#linear_model = tf.layers.Dense(units=1)
#y = linear_model(x)

