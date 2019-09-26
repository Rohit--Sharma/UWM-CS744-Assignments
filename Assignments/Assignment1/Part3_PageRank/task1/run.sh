#!/bin/bash
set -o noglob
spark-submit pagerank.py $1 $2 $3 $4 $5 $6
