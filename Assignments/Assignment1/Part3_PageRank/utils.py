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

