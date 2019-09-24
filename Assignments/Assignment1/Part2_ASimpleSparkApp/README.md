# Part 2: A Simple Spark Application

The application sorts a csv file alphabetically based on two columns: first based on country code (third column) and then by timestamp (last column). We implemented a Spark program in Python called `sorting_with_spark.py` to do the same. It first loads the csv file as an RDD and then applies `sortBy` transformation which takes the third and last columns as keys. In the next step, it saves the output file using the `saveAsTextFile` action.

## Usage:
To execute the program, navigate to the directory with the python scripts, ensure that you have added `/spark-2.4.4-bin-hadoop2.7/bin` to your path, and follow the syntax below:
```
$ ./run.sh <input_filename> <output_filename>
```
