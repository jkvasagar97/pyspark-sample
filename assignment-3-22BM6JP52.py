from pyspark import SparkContext
import re
import sys
from itertools import combinations
from math import log2
#import time

# Util function to split a line into lowercase words
def split_line(line):
    line = re.split('[^a-zA-Z0-9]+', line.lower())
    return [word for word in line if word!='']

# Util function to generate word pairs from a list of words
def generate_word_pairs(words):
    return list(combinations(words, 2))

# Util function to calculate pmi
def gen_pmi(combined_count, word1_count, word2_count, docs_count, pair):
    return (pair,log2(docs_count * combined_count/(word1_count * word2_count)))

if __name__ == '__main__':
    #start = time.time()
    file_path = sys.argv[1]
    query_word = sys.argv[2]
    k = sys.argv[3]
    stop_word_file = sys.argv[4]
    sc = SparkContext('local', 'PMI')
    
    # Read input file as an RDD of lines
    lines_rdd = sc.textFile(file_path)
    total_docs = lines_rdd.count()

    # Read and store stop words
    with open(stop_word_file, 'r') as file:
        stop_words = file.read()
        stop_words = stop_words.split('\n')

    broadcastVar = sc.broadcast((stop_words, query_word.lower(), k, total_docs))
    
    # Split lines into lowercase words and count each word
    word_count_rdd = lines_rdd.flatMap(split_line) \
                            .filter(lambda word: word not in broadcastVar.value[0])\
                            .map(lambda word: (word, 1)) \
                            .reduceByKey(lambda x, y: x+y)

    # count of query words
    query_word_count = word_count_rdd.filter(lambda word: word[0] == query_word).collect()
    query_word_count = sc.broadcast(query_word_count)

    # Creating word pairs
    pair_count_rdd= lines_rdd.map(split_line)\
            .filter(lambda word: query_word in word)\
            .map(generate_word_pairs)\
            .flatMap(lambda x: x).filter(lambda x: query_word in x)\
            .filter(lambda x: x[0] not in broadcastVar.value[0] and x[1] not in broadcastVar.value[0])\
            .map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)

    # Getting the list of owords which form pair with query word
    other_words_rdd = pair_count_rdd.map(lambda pair: pair[0][1] if pair[0][0]== broadcastVar.value[1] else pair[0][0])
    other_word_list = other_words_rdd.collect()
    other_word_count = word_count_rdd.filter(lambda word: word[0] in other_word_list)
    other_word_count = dict(other_word_count.collect())
    other_word_count[broadcastVar.value[1]] = query_word_count.value[0][1]

    # Take only top -ve or 0 values
    bot_k_rdd = pair_count_rdd.map(lambda pair: gen_pmi(pair[1], other_word_count[pair[0][0]], 
                                                        other_word_count[pair[0][1]], 
                                                        broadcastVar.value[3], pair[0])).filter(lambda x: x[1] <= 0)\
                                                            .takeOrdered(int(broadcastVar.value[2]), lambda x:x[1])
                                                        
    # Take only top +ve or 0 values
    top_k_rdd = pair_count_rdd.map(lambda pair: gen_pmi(pair[1], other_word_count[pair[0][0]], 
                                                        other_word_count[pair[0][1]], 
                                                        broadcastVar.value[3], pair[0])).filter(lambda x: x[1] >= 0)\
                                                            .takeOrdered(int(broadcastVar.value[2]), lambda x:-x[1])
                                                        

    print("Top k or +ve PMIs", top_k_rdd)
    print("Top k or -ve PMIs", bot_k_rdd)
    #print(time.time()-start)