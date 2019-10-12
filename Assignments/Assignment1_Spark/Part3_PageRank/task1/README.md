# Task 1: Implementing PageRank in Python Spark

This application implements the PageRank algorithm in Python Spark, using a simplified algorithm. The initial ranks of each page are first set to 1, and then on each iteration, each page's rank is updated. Our implementation of PageRank runs for 5 iterations. 
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that you have added `spark-2.4.4-bin-hadoop2.7/bin` to your path, and follow the syntax below:

```
$ ./run.sh <small/large> <num_iterations> <num_partitions> <spark_master_hostname> <input_path> <output_path> 

```
### Example 
```
$ ./run.sh large 5 96 c220g1-031124vm-1.wisc.cloudlab.us /proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/*xml* pagerank_task1_output 
```
