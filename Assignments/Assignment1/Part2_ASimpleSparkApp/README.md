# A Simple Spark Application

The application sorts a csv file alphabetically based on two columns: first based on country code (third column) and then by timestamp (last column). We implemented a Spark program in Python called `sorting_with_spark.py` to do the same. It first loads the csv file as an RDD and then applies `sortBy` transformation which takes the third and last columns as keys. In the next step, it saves the output file using the `saveAsTextFile` action.

## Usage:
To execute the program, follow the syntax below:
```
$ spark-submit sorting_with_spark.py <input_file> <output_file>
```
