import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# TF graph input
x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model hyperparameters
learning_rate = 0.01
display_step = 1
batch_size = 100
num_iter = 50

# logistic regression functions
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# adding loss summary
tf.summary.scalar("loss", loss)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # putting each tensorboard log into its own dir
    now = time.time()
    writer = tf.summary.FileWriter("./tmp/mnist_logs/{}".format(now), sess.graph_def)

    sess.run(init)

    for iter in range(num_iter):

        avg_loss = 0
        num_batches = int(mnist.train.num_examples/batch_size)

        for i in range(num_batches):
            data_x, data_y = mnist.train.next_batch(batch_size)
            c, summ = sess.run([loss, merged], feed_dict={x: data_x, y: data_y})

            # getting average loss per iteration
            avg_loss += c / num_batches
            writer.add_summary(summ, iter * num_batches + i)

        # printing the avg loss after every iteration (epoch)
        if (iter+1) % display_step == 0:
            print("Epoch:", '%04d' % (iter+1), "cost=", "{:.9f}".format(avg_loss))

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    # Calculate accuracy on test data
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

writer.flush()
writer.close()