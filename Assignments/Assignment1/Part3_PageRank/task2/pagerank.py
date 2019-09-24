from operator import add
from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("Part3_PageRank")\
	.set('spark.executor.memory', '29g')\
	.set('spark.executor.cores', '5')\
	.set('spark.driver.memory', '29g')\
	.set('spark.task.cpus', '1')\
	.setMaster('spark://' + 'c220g1-031124vm-1.wisc.cloudlab.us' + ':7077')
sc = SparkContext(conf=conf)

def url_partitioner(url):
    return hash(url)


#documents = sc.textFile("/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/")
documents = sc.textFile("web-BerkStan.txt")
n_iter = 10

# Filter the comments beginning with # and create an RDD 
links = documents.filter(lambda x: x[0] != '#').map(lambda x: (x.split('\t')[0], x.split('\t')[1]))
num_partitions = links.getNumPartitions()
print("******************************************** Number of partitions : %d" % num_partitions)

links = links.groupByKey().mapValues(list).partitionBy(num_partitions, url_partitioner)

ranks = links.mapValues(lambda x: 1).partitionBy(num_partitions, url_partitioner)

def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)

for i in range(n_iter):
    # Build an RDD of (targetURL, float) pairs
    # with the contributions sent by each page
    contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]), True)
    # Sum contributions by URL and get new ranks
    ranks = contribs.reduceByKey(add, num_partitions, url_partitioner).mapValues(lambda sum: 0.15 + 0.85 * sum)

# ranks = ranks.sortBy(lambda url_rank: url_rank[1], ascending=False)

ranks_ = ranks.collect()
# print(ranks_[0:100])
