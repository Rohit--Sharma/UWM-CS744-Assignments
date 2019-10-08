#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh

while true
do
    echo "Please select a mode: Single: s, Distributed-synchronous: sync, Distributed-asynchronous: async"
    read option
    if [ "$option" = "s" ] || [ "$option" = "sync" ] || [ "$option" = "async" ]; then
        break
    else
        echo "Please enter a valid mode."
    fi
done

if [ "$option" = "sync" ] || [ "$option" = "async" ]; then
    while true
    do
        echo "Please select the cluster: cluster or cluster2"
        read cluster
        if [ "$cluster" = "cluster" ] || [ "$cluster" = "cluster2" ]; then
            break
        else
            echo "Please enter a valid cluster."
        fi
    done
fi

if [ "$option" = "s" ]; then
    start_cluster code_template.py single
elif [ "$option" = "async" ]; then
    start_cluster logistic_regression_async.py "$cluster"
elif [ "$option" = "sync" ]; then
    start_cluster logistic_regression_sync.py "$cluster"
fi
