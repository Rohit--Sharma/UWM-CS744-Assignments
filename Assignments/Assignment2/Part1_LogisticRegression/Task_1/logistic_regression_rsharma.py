import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# TF graph input
x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initializing the variables
# init = tf.initialize_all_variables()

prediction  = tf.nn.softmax(tf.matmul(x, W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()

with tf.Session() as sess:
    sess.run(init)
    batch_size = 1
    num_iter = 100

    data_x, data_y = mnist.train.next_batch(batch_size)
    for iter in range(num_iter):
        _, loss_val = sess.run((train, loss), feed_dict={x: data_x, y: data_y})
        print(loss_val)
        tf.summary.scalar('loss', loss_val)

writer.close()
