from pyspark.sql.functions import col
from pyspark import SparkContext, SparkConf 


conf = SparkConf().setAppName("Part2_SortingWithSpark").setMaster("local")
sc = SparkContext(conf=conf)


dataframe = sc.textFile("export.csv")

dataframe.sortByKey('cca2')