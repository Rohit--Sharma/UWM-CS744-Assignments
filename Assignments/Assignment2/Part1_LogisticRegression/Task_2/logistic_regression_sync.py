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
    clusterinfo = clusterSpec[FLAGS.deploy_mode]
    server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo)):

            is_chief = (FLAGS.task_index == 0)
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

            sync_init_op = optimizer.get_init_tokens_op()
            chief_queue_runner = optimizer.get_chief_queue_runner()

            # Initialising the variables.
            init_op = tf.global_variables_initializer()
            
            sv = tf.train.Supervisor(is_chief=is_chief,
                logdir="/tmp/train_logs",
                init_op=init_op,
                global_step=global_step,
                save_model_secs=600)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                sess.run(init_op)
                if is_chief:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    # Insert initial tokens to the queue.
                    sess.run(sync_init_op)
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
