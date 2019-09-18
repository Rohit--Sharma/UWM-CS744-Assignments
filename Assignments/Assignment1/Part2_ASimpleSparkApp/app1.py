from pyspark.sql.functions import col
from pyspark import SparkContext, SparkConf 


conf = SparkConf().setAppName("Part2_SortingWithSpark").setMaster("local")
sc = SparkContext(conf=conf)


dataframe = sc.textFile("export.csv")

sorted_dataframe = dataframe.sortBy(lambda x: (x.split(",")[2], x.split(",")[-1]))

print(sorted_dataframe.collect())