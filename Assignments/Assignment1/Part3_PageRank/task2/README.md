# Task 2: Adding custom RDD partitioning to PageRank

In this task, we've added custom RDD partitioning to our previous implementation of PageRank. We're partitioning by hash code, and using the same partitioner for both the links and the ranks.
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that you have added `spark-2.4.4-bin-hadoop2.7/bin` to your path, and follow the syntax below:
```
$ ./run.sh <small/large>
```
