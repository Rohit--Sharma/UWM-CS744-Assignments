# Task 2: Adding custom RDD partitioning to PageRank

In this task, we've added custom RDD partitioning to our previous implementation of PageRank. We're partitioning by hash code, and using the same partitioner for both the links and the ranks.
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts and ensure that you have added `spark-2.4.4-bin-hadoop2.7/bin` to your path or have spark mounted in `/mnt/data/`. Then follow the syntax below:
```
$ ./run.sh <small/large> <num_iterations> <num_partitions> <spark_master_hostname> <input_path> <output_path> 

```
### Example 
```
$ ./run.sh large 5 96 c220g1-031124vm-1.wisc.cloudlab.us "/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/*xml*" pagerank_task2_output
```
