from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("Part3_PageRank").setMaster("local")
sc = SparkContext(conf=conf)

#documents = sc.textFile("/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/")
documents = sc.textFile("web-BerkStan.txt")

ranks = lines.groupByKey().mapValues(lambda x : 1)

ITERATIONS = 10
for i in to range(ITERATIONS):
	#Build an RDD of (targetURL, float) pairs // with the contributions sent by each page 
	contribs = links.join(ranks).flatMap(lambda url_rank : links.map(lambda url_rank : (url_rank[1][0], url_rank[1][1]/len(url_rank[1]))))
	#Sum contributions by URL and get new ranks 
	ranks = contribs.reduceByKey(lambda x, y : x+y).mapValues(lambda r :  0.15 + 0.85 * r)