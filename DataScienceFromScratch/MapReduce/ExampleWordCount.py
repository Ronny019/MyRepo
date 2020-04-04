import Implementation as imp
from collections import defaultdict

def word_count_old(documents):
    """word count not using MapReduce"""
    return Counter(word
                    for document in documents
                    for word in imp.tokenize(document))

def wc_mapper(document):
    """for each word in the document, emit (word,1)"""
    for word in imp.tokenize(document):
        yield (word, 1)


def wc_reducer(word, counts):
    """sum up the counts for a word"""
    yield (word, sum(counts))


def word_count(documents):
    """count the words in the input documents using MapReduce"""
    # place to store grouped values
    collector = defaultdict(list)
    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.items()
            for output in wc_reducer(word, counts)]


documents = ["data science", "big data", "science fiction"]

wc_mapper_results = [result
                     for document in documents
                     for result in wc_mapper(document)]

print("wc_mapper results")
print(wc_mapper_results)
print()
print("word count results")
print(word_count(documents))
print()