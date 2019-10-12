# Part 2: Distributed LeNet using Keras

LeNet model is implemented using Keras API with TensorFlow backend. The code is implemented in a distributed manner using MultiWorkerMirroredStrategy API and tested on clusters of size 1, 2 and 3 nodes.

## Usage

To execute the program, navigate to the directory with the python scripts, and follow the syntax below:

```bash
$ sudo bash run.sh <batch_size> <num_epochs> <cluster_mode>
```

Here, the `cluster_mode` is one of `single`, `cluster` or `cluster2` to run it on 1, 2, or 3 nodes respectively.

## Terminating the cluster

To terminate the cluster before the next run, run the following:

```bash
$ sudo bash terminate_cluster.sh
```
