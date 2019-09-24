from operator import add
from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("Part3_PageRank").setMaster("local")
sc = SparkContext(conf=conf)

documents = sc.textFile("/proj/uwmadison744-f19-PG0/data-part3/enwiki-pages-articles/*xml*")

def filter_func(x):
	if ":" in x and not x.startswith("category:"):
		return False
	return True	

def split_and_case_func(x):
        temp = x.lower().split('\t')
        return (temp[0], temp[1])

n_iter = 5

# Filter the comments beginning with # and create an RDD 
links = documents.filter(lambda x: filter_func(x[1])).map(lambda x: split_and_case_func(x)).groupByKey()
ranks = links.mapValues(lambda x: 1)

def computeContribs(urls, rank):
	"""Calculates URL contributions to the rank of other URLs."""
	num_urls = len(urls)
	for url in urls:
		yield (url, rank / num_urls)

for i in range(n_iter):
	# Build an RDD of (targetURL, float) pairs
	# with the contributions sent by each page
	contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))
	# Sum contributions by URL and get new ranks
	ranks = contribs.reduceByKey(add).mapValues(lambda sum: 0.15 + 0.85 * sum)

# ranks = ranks.sortBy(lambda url_rank: url_rank[2], ascending=False)

ranks_ = ranks.count() #saveAsTextFile("/mnt/data/result.txt")
#print(ranks_[1:100])
