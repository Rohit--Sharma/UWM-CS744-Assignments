from pyspark.sql.functions import col
from pyspark import SparkContext, SparkConf 
import sys

input_filename = sys.argv[1]
output_filename = sys.argv[2]

conf = SparkConf().setAppName("Part2_SortingWithSpark").setMaster("local")
sc = SparkContext(conf=conf)


dataframe = sc.textFile(input_filename)

sorted_dataframe = dataframe.sortBy(lambda x: (x.split(",")[2], x.split(",")[-1]))

sorted_dataframe.saveAsTextFile(output_filename)