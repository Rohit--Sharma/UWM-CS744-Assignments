# Part 1: Logistic Regression using Tensorflow

Logistic Regression model is implemented using Tensorflow backend.

For the distributed mode(Task 2) code is implemented using MonitoredTrainingSession for Synchronous and Asynchronous SGD and tested on a cluster of size 1, 2 and 3 nodes. SyncReplicasOptimizer wrapper is used for Sync SGD.

## Usage for Task 1

To execute the program, navigate to the directory with the python scripts, and follow the syntax below:

```bash
$ sudo bash run.sh
```

## Usage for Task 2

To execute the program, navigate to the directory with the python scripts, and follow the syntax below:

```bash
$ sudo bash run.sh 
Please select a mode: Single: s, Distributed-synchronous: sync, Distributed-asynchronous: async
<s/sync/async>
Please select the cluster: cluster or cluster2
<cluster_mode>
``` 

Here, the `cluster_mode` is one of `single`, `cluster` or `cluster2` to run it on 1, 2, or 3 nodes respectively.


## Terminating the cluster for Task 2

To terminate the cluster before the next run, run the following:
```bash
$ sudo bash terminate_cluster.sh
```
