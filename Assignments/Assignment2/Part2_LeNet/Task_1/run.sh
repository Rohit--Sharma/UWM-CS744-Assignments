#!/bin/bash

# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh

start_cluster lenet1.py cluster
start_cluster lenet2.py cluster
start_cluster lenet3.py cluster
