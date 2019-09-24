if [ $1 = "small" ]; then
	spark-submit pagerank.py
elif [ $1 = "large" ] ; then
	spark-submit pagerank_enwiki.py
else
	echo "Invalid argument. Please enter either \"small\" or \"large\"."
fi