import sys
from operator import add
from pyspark import SparkContext, SparkConf

file_type = sys.argv[1] 
n_iter = int(sys.argv[2])
num_partitions = int(sys.argv[3])
spark_master_hostname = sys.argv[4]
input_path = sys.argv[5]
output_path = sys.argv[6]

def filter_func(x, file_type):
	if file_type == "small":
		return x[0] != '#'
	else:
		if ":" in x[1] and not x[1].startswith("category:"):
			return False
		return True

def split_func(x, file_type):
	if file_type == "small":
		temp = x.split('\t')
		return (temp[0], temp[1])
	else:
		temp = x.lower().split('\t')
		return (temp[0], temp[1])

def computeContribs(urls, rank):
	"""Calculates URL contributions to the rank of other URLs."""
	num_urls = len(urls)
	for url in urls:
		yield (url, rank / num_urls)

conf = SparkConf().setAppName("Part3_PageRank")\
	.set('spark.executor.memory', '29g')\
	.set('spark.executor.cores', '5')\
	.set('spark.driver.memory', '29g')\
	.set('spark.task.cpus', '1')\
	.setMaster('spark://' + spark_master_hostname + ':7077')
sc = SparkContext(conf=conf)

documents = sc.textFile(input_path)

# Filter the comments beginning with # and create an RDD 
links = documents.filter(lambda x: filter_func(x, file_type)).map(lambda x: split_func(x, file_type)).groupByKey().partitionBy(num_partitions)

# Initialise the rank of each to 1
ranks = links.mapValues(lambda x: 1).partitionBy(num_partitions)

# On each iteration, each page contributes to its neighbors by rank(p)/ # of neighbors.
for i in range(n_iter):
	# Build an RDD of (targetURL, float) pairs
	# with the contributions sent by each page
	contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

	# Sum contributions by URL and get new ranks
	ranks = contribs.reduceByKey(add).mapValues(lambda sum: 0.15 + 0.85 * sum).partitionBy(num_partitions)


ranks_ = ranks.saveAsTextFile(output_path)