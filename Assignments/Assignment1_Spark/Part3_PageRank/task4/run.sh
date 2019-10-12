#!/bin/bash
set -o noglob
/mnt/data/spark-2.4.4-bin-hadoop2.7/bin/spark-submit pagerank.py $1 $2 $3 $4 $5 $6
