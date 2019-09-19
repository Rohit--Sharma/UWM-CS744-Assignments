from operator import add
from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("Part3_PageRank").setMaster("local")
sc = SparkContext(conf=conf)

#documents = sc.textFile("/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/")
documents = sc.textFile("web-BerkStan.txt")
n_iter = 10

# Filter the comments beginning with # and create an RDD 
links = documents.filter(lambda x: x[0] != '#').map(lambda x: (x.split('\t')[0], x.split('\t')[1])).persist()

ranks = links.groupByKey().mapValues(lambda x: 1)

links = links.groupByKey().mapValues(len)

for i in range(n_iter):
    # Build an RDD of (targetURL, float) pairs
    # with the contributions sent by each page
    contribs = links.join(ranks).flatMap(lambda x: x[1][0].map(lambda dest: (dest, x[1][1] / len(x[1][0]))))
    # Sum contributions by URL and get new ranks
    ranks = contribs.reduceByKey(add).mapValues(lambda sum: 0.15 + 0.85 * sum)

ranks.collect()
