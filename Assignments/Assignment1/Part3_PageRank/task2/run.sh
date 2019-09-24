if [ $1 = "small" ]; then
	/mnt/data/spark-2.4.4-bin-hadoop2.7/bin/spark-submit pagerank.py
elif [ $1 = "large" ] ; then
	/mnt/data/spark-2.4.4-bin-hadoop2.7/bin/spark-submit pagerank_enwiki.py
else
	echo "Invalid argument. Please enter either \"small\" or \"large\"."
fi