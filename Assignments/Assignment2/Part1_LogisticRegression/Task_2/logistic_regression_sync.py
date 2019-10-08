import time
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
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps": [
        "node0:2222"
    ],
    "worker": [
        "node0:2223",
        "node1:2222",
        "node2:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

def main():
    clusterinfo = clusterSpec[FLAGS.deploy_mode]
    server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # Configure
    config=tf.ConfigProto(log_device_placement=False)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        # model hyperparameters
        learning_rate = 0.01
        display_step = 1
        batch_size = 75
        num_iter = 100000
        is_chief = (FLAGS.task_index == 0)
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo)):

            # TF graph input
            x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784
            y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

            # Set model weights
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            # logistic regression prediction and lossfunctions
            prediction = tf.nn.softmax(tf.matmul(x, W) + b)
            loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))

            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            global_step = tf.contrib.framework.get_or_create_global_step()

            # Gradient Descent
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2,
                                   total_num_replicas=2)
            
            training_op = optimizer.minimize(loss, global_step=global_step)
            hooks = [optimizer.make_session_run_hook(is_chief, num_tokens=0)]
            # tf.train.StopAtStepHook(last_step=num_iter), 
            # adding loss summary
            loss = tf.summary.scalar("loss", loss)
            merged = tf.summary.merge_all()

            mon_sess = tf.train.MonitoredTrainingSession(
                master=server.target, 
                is_chief=is_chief,
                config=config,
                hooks=hooks,
                stop_grace_period_secs=10,
                checkpoint_dir="/tmp/train_logs")
            iter = 0
            
            # putting each tensorboard log into its own dir
            now = time.time()
            writer = tf.summary.FileWriter("./tmp/mnist_logs/{}".format(now))

            while not mon_sess.should_stop():
                data_x, data_y = mnist.train.next_batch(batch_size)
                _, loss_val, summ = mon_sess.run((training_op, loss, merged), feed_dict={x: data_x, y: data_y})

                writer.add_summary(summ, iter)
                if (iter+1) % display_step == 0:		
                    print("Epoch:", '%04d' % (iter+1), "cost=", "{:.9f}".format(loss_val))
                iter += 1
            print('Done',FLAGS.task_index)

if __name__ == "__main__":
    time_begin = time.time()
    main()
    time_end = time.time()

    training_time = time_end - time_begin
    print('Total time taken:', training_time, 's')
