# Task 3: Persisting RDDs in memory

In this task, we're persisting the links RDD in memory so that operations done on it in each iteration take a shorter amount of time. 
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that spark-submit is located at `/mnt/data/spark-2.4.4-bin-hadoop2.7/bin/spark-submit` and follow the syntax below:
```
$ ./run.sh <small/large> 
```
