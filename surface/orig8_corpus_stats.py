
import random
import pandas as pd
import compare_bigrams
import sys
import os
import as_numeric

import quantify_copying


# The file that contains the base corpus
INPUT_FILE = "../corpus/cath8.txt"
# Each line should be a sentence, with the words separated by spaces


# Read the input file and obtain a list of list of strings, i.e. the list of sentences
f = open(INPUT_FILE,'r')
lines = f.readlines()
f.close()
sentences = [ l.strip().split(" ") for l in lines ]


# Get a list of the bigrams
unigrams_unpermuted = compare_bigrams.unigram_counts( sentences )
bigrams_unpermuted  = compare_bigrams.bigram_counts( sentences )



#print "The corpus has %i words, originally in %s sentences"%(len(flat_sentences),len(sentences))


# First, put all the words in a line

corpus_stats = pd.DataFrame()


if True:

    corpus = sentences[:] # make a copy of cath8, then compare it to itself
    
    # Count the bigrams of our permuted corpus
    bigrams_permuted = compare_bigrams.bigram_counts( corpus )

    # Get some statistics on the comparison of bigrams
    generated_corpus_bigrams  = compare_bigrams.bigram_counts ( corpus )
    generated_corpus_unigrams = compare_bigrams.unigram_counts( corpus )
    
    comp = compare_bigrams.compare_Ngrams( bigrams_unpermuted, generated_corpus_bigrams )
    comp = dict([ ("bigrams.%s"%k,v) for (k,v) in comp.items() ])
    comp["permutation"]=0
    
    unicomp = compare_bigrams.compare_Ngrams( unigrams_unpermuted, generated_corpus_unigrams )
    unicomp = dict([ ("unigrams.%s"%k,v) for (k,v) in unicomp.items() ])

    comp = {**comp,**unicomp} # merge the dicts
    comp["n.unique.bigrams"]  = len(generated_corpus_bigrams.keys())
    comp["n.unique.unigrams"] = len(generated_corpus_unigrams.keys())


    if True:

        # Quantify the copying
        cop = quantify_copying.corpus(corpus)

        corpus_stats = pd.concat([corpus_stats,
                                  pd.DataFrame({**comp,**cop},
                                               index=[1])])


corpus_stats.to_csv('interim/cath8_stats.csv')
