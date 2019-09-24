# Task 4: Killing worker processes

In this task, we're running the PageRank implementation used in task 1 and killing a worker process when the application reaches 25% and 75% of its lifetime. 
To choose whether to run PageRank on the smaller dataset of the Berkeley-Stanford web graph, type "small" as a command line argument. If you'd like to use the dataset of enwiki-pages-articles, type "large" as a command line argument. 

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that you have added `spark-2.4.4-bin-hadoop2.7/bin` to your path, and follow the syntax below:
```
$ ./run.sh <small/large> 
```
