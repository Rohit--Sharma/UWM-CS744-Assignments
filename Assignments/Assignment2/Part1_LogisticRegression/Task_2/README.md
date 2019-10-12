# Task 2: Distributed Logistic Regression using Tensorflow

Logistic Regression model is implemented using Tensorflow backend. The code is implemented in a distributed manner using MonitoredTrainingSession for Synchronous and Asynchronous SGD and tested on a cluster of size 1, 2 and 3 nodes. SyncReplicasOptimizer wrapper is used for Sync SGD.

## Usage

To execute the program, navigate to the directory with the python scripts, and follow the syntax below:

```bash
$ sudo bash run.sh 
Please select a mode: Single: s, Distributed-synchronous: sync, Distributed-asynchronous: async
<s/sync/async>
Please select the cluster: cluster or cluster2
<cluster_mode>
``` 

Here, the `cluster_mode` is one of `single`, `cluster` or `cluster2` to run it on 1, 2, or 3 nodes respectively.


## Terminating the cluster

To terminate the cluster before the next run, run the following:
```bash
$ sudo bash terminate_cluster.sh
```
