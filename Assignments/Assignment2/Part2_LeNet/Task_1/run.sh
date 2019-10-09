#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh

while true
do
    echo "Please select a mode: Single: single, Distributed-synchronous (2 nodes): cluster, Distributed-synchronous (3 nodes): cluster2"
    read option
    if [ "$option" = "single" ] || [ "$option" = "cluster" ] || [ "$option" = "cluster2" ]; then
        break
    else
        echo "Please enter a valid mode."
    fi
done

start_cluster lenet.py "$option"
