from pyspark import SparkContext, SparkConf 

# The first thing that Spark application does is creating an object SparkContext.
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)

# You can read the data from file into RDD object as a collection of lines
lines = sc.textFile("data.txt")

lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)
