# Task 4: Killing worker processes

In this task, we're running the PageRank implementation used in task 1 and killing a worker process when the application reaches 25% and 75% of its lifetime. 
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that you have added `spark-2.4.4-bin-hadoop2.7/bin` to your path, and follow the syntax below:
```
$ ./run.sh <small/large> <num_iterations> <num_partitions> <spark_master_hostname> <input_path> <output_path> 
```

## Example
```
$ ./run.sh large 5 96 c220g1-031124vm-1.wisc.cloudlab.us /proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/*xml* pagerank_task3_output 
```
After triggering the process, we killed the worker at 25% and 75% of the lifetime. To do the same, we first cleared the memory cache using `sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"` and then we killed the worker process.
