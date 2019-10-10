#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh

if [ "$#" -ne 3 ]; then
    echo "Illegal arguments passed. Please enter the input in the below format:"
    echo "run.sh <batch_size> <num_epochs> <cluster_mode>"
    echo "The mode is one of single, cluster or cluster2"
    exit 2
fi

start_cluster lenet.py $1 $2 $3
