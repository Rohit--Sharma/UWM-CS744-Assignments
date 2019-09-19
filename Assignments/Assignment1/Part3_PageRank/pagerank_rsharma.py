from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("Part3_PageRank").setMaster("local")
sc = SparkContext(conf=conf)

#documents = sc.textFile("/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/")
documents = sc.textFile("web-BerkStan.txt")

# Filter the comments beginning with # and create an RDD 
lines = documents.filter(lambda x: x[0] != '#').map(lambda x: (x.split('\t')[0], x.split('\t')[1]))

