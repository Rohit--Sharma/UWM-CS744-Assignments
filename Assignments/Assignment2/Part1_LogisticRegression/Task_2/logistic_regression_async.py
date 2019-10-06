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
        "localhost:2222"
    ],
    "worker" : [
        "10.10.1.2:2223",
        "10.10.1.3:2222"
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


def main(_):
    # Create a cluster from the parameter server and worker hosts.
    clusterinfo = clusterSpec[FLAGS.deploy_mode]

    # Create and start a server for the local task.
    server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Loading dataset
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

        #model hyperparameters
        learning_rate = 0.01
        display_step = 1
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

            global_step = tf.contrib.framework.get_or_create_global_step()
            
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=num_iter)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=(FLAGS.task_index == 0),
                                            checkpoint_dir="~/train_logs",
                                            hooks=hooks) as mon_sess:
            mon_sess.run(init)

            iter = 0
            while not mon_sess.should_stop():
                data_x, data_y = mnist.train.next_batch(batch_size)

                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                _, loss_val = mon_sess.run((optimizer, loss), feed_dict={x: data_x, y: data_y})

                # printing the loss after every iteration (epoch)
                if (iter+1) % display_step == 0:
                    print("Epoch:", '%04d' % (iter+1), "cost=", "{:.9f}".format(loss_val))
                iter += 1


if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])
