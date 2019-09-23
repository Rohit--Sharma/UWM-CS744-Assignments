import sys
from pyspark import SparkContext, SparkConf


# Input and output file name from the command line
input_filename = sys.argv[1]
output_filename = sys.argv[2]

# Creating a Spark context object using the configuration specified below
conf = SparkConf().setAppName("Part2_SortingWithSpark").setMaster("local")
sc = SparkContext(conf=conf)

# Load the input file as an RDD
# TODO: Load the data into HDFS and load the file from HDFS using "hdfs://..."
lines = sc.textFile(input_filename)
header = lines.first()
filtered_lines = lines.filter(lambda line: line != header)

# Sort based on country code (3rd column) first and then timestamp (last column) with sortBy transformation
sorted_lines = filtered_lines.sortBy(lambda x: (x.split(",")[2], x.split(",")[-1]))

# add the header as an rdd and join it to the sorted rdd
full = sc.parallelize([header, ""])
full = full.filter(lambda x: x != "").union(sorted_lines)

# Write the output to disk with saveAsTextFile action
full.coalesce(1, True).saveAsTextFile(output_filename)