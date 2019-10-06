from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

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
        "host_name0:2223",
        "host_name1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "host_name0:2222"
    ],
    "worker" : [
        "host_name0:2223",
        "host_name1:2222",
        "host_name2:2222",
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

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo)):
            # TF graph input
            x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784
            y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

            # Set model weights
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            #model hyperparameters
            learning_rate = 0.01
            display_step = 1
            batch_size = 100
            num_iter = 50

            # logistic regression functions
            prediction = tf.nn.softmax(tf.matmul(x, W) + b)
            loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))

            global_step = tf.contrib.framework.get_or_create_global_step()

            optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2,
                                   total_num_replicas=2)

            #init_token_op = opt.get_init_tokens_op()
            #chief_queue_runner = opt.get_chief_queue_runner()


            # Initialising the variables.
            init = tf.global_variables_initializer()
            # The StopAtStepHook handles stopping after running given steps.
            hooks=[tf.train.StopAtStepHook(last_step=1000000)]
            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            sv = tf.train.Supervisor(is_chief=is_chief,
                logdir="/tmp/train_logs",
                init_op=init_op,
                global_step=global_step,
                save_model_secs=600)

            with training.MonitoredTrainingSession(
                master=workers[worker_id].target, is_chief=is_chief,
                hooks=[sync_replicas_hook]) as mon_sess:
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
    main()
