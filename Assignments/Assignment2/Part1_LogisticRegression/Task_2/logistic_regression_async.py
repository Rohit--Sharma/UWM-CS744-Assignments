import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node0.rsharma-assign2.uwmadison744-f19-PG0.wisc.cloudlab.us:2222"
    ],
    "worker" : [
        "node0.rsharma-assign2.uwmadison744-f19-PG0.wisc.cloudlab.us:2223",
        "node1.rsharma-assign2.uwmadison744-f19-pg0.wisc.cloudlab.us:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "localhost:2222"
    ],
    "worker" : [
        "localhost:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}


def main():
    # Create a cluster from the parameter server and worker hosts.
    clusterinfo = clusterSpec[FLAGS.deploy_mode]

    # Create and start a server for the local task.
    server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    print('Job name:', FLAGS.job_name)

    if FLAGS.job_name == "ps":
        print("Executing PS job")
        server.join()
    elif FLAGS.job_name == "worker":
        print("Executing Worker job")

        # Loading dataset
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

        #model hyperparameters
        learning_rate = 0.01
        display_step = 50
        batch_size = 100
        num_iter = 500

        # Ref: https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md#distributed-tensorflow
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):
            print("Starting a job with task id:", FLAGS.task_index)

            # TF graph input
            x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784
            y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

            # Set model weights
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            # Build model...
            prediction = tf.nn.softmax(tf.matmul(x, W) + b)
            loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=1))

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session(server.target) as sess:
            sess.run(init)

            for iter in range(num_iter):
                avg_loss = 0
                num_batches = int(mnist.train.num_examples/batch_size)

                for _ in range(num_batches):
                    data_x, data_y = mnist.train.next_batch(batch_size)
                    _, loss_val = sess.run((optimizer, loss), feed_dict={x: data_x, y: data_y})

                    avg_loss += loss_val / num_batches

                # printing the loss after every iteration (epoch)
                if (iter+1) % display_step == 0:
                    print("Epoch:", '%04d' % (iter+1), "loss=", "{:.9f}".format(avg_loss))

            # Computing the model accuracy
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            # Calculate accuracy on test data
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if __name__ == "__main__":
    main()
