from pyspark import SparkContext, SparkConf 

# The first thing that Spark application does is creating an object SparkContext.
conf = SparkConf().setAppName("Part2_SortingWithSpark").setMaster("local")
sc = SparkContext(conf=conf)

# You can read the data from file into RDD object as a collection of lines
lines = sc.textFile("export.csv")

# Sort based on 3rd column (country code)
linesSorted = lines.sortBy(lambda a: a.split(',')[2])

linesSorted.saveAsTextFile("export_sorted.csv")
