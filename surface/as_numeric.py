


# This module can convert the corpus to numeric values,
# based on the vocabulary list.

f = open('../corpus/Sigma8.txt')
vocab = [ w.strip() for w in f.readlines() ]
f.close()





def corpus_to_numeric( corpus ):
    # Given a list of a list of words, output the corpus as numeric
    # values. For simplicity, convert them to strings.
    return [ [ "%i"%(vocab.index(w)+1) for w in s ] for s in corpus ]

