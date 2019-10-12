import sys
from operator import add
from pyspark import SparkContext, SparkConf

file_type = sys.argv[1] 				# "small/large"
n_iter = int(sys.argv[2])				# Number of iterations for page rank algorithm
num_partitions = int(sys.argv[3])		# Num of RDD Paritions to create
spark_master_hostname = sys.argv[4]		# Master URL for a distributed cluster
input_path = sys.argv[5]				# Path to input files
output_path = sys.argv[6]				# Path to save the output. (pagerank for each url)

def filter_func(x, file_type):
	"""
	The input data is filtered based on the filetype. 
	For "small" file_type, if the input contains a comment i.e. if x contains "#" it returns False, else 
	 it return True.
	For "large " file_type, if the destination article contains a ':', we ignore the entire link unless
	 it starts with 'Category:'(returns False), else it returns True.
	:param x: string, string based on which filtering is performed.
	:param file_type: string, small/large. 
	"""
	if file_type == "small":
		return x[0] != '#'
	else:
		if ":" in x[1] and not x[1].startswith("category:"):
			return False
		return True

def split_func(x, file_type):
	"""
	Splits the input into two. If the filetype is large it also converts the input into lower case.
	:param x: string, string based on which filtering is performed.
	:param file_type: string, small/large.
	"""
	if file_type == "small":
		temp = x.split('\t')
		return (temp[0], temp[1])
	else:
		temp = x.lower().split('\t')
		return (temp[0], temp[1])

def computeContribs(urls, rank):
	"""
	Calculates rank contributions to URLs.
	:param urls: URLs for which rank is calculated.
	:param rank: Rank which is distributed to all the URLs.
	"""
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

# Read input file(s)
documents = sc.textFile(input_path)

# Filter the input, transform the input into (key-value) pairs with source article as the key and destination article as its value, and group by 
# source to generate (URL, outlinks) pairs.
links = documents.filter(lambda x: filter_func(x, file_type)).map(lambda x: split_func(x, file_type)).groupByKey().partitionBy(num_partitions).cache()

# Initialise the rank of each to 1
ranks = links.mapValues(lambda x: 1).partitionBy(num_partitions)


for i in range(n_iter):
	"""
	On each iteration, each page contributes to its neighbors by rank(p)/# of neighbors.
	:param n_iter: Number of iterations
	"""
	
	# Build an RDD of (targetURL, float) pairs with the contributions sent by each page
	contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

	# Sum contributions by URL and get new ranks
	ranks = contribs.reduceByKey(add).mapValues(lambda sum: 0.15 + 0.85 * sum).partitionBy(num_partitions)

# Write the output to output_path as a text file.
ranks_ = ranks.saveAsTextFile(output_path)