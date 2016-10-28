""" This cute little script takes a corpus and outputs any number of shuffled versions.
The shuffled (permuted) versions are calculated by permuting all words within each sentence.

In this way, we get a randomised corpus that has the same unigram frequencies, sentence lengths and corpus size.

"""


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

OUTPUT_DIR = "./output/permutations_within_sentences/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Number of permutations
N = 1000

# The size of the train corpus (in number of sentences)
#N_TRAIN = 184


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



i=0
n_tries=0
while i<N:
    #for i in range(N):

    # Make a copy of the sentence corpus
    corpus = [ s[:] for s in sentences ]


    for s in corpus:
        random.shuffle(s) # should actually permute the sentence because pass-by-reference


    
    # Count the bigrams of our permuted corpus
    bigrams_permuted = compare_bigrams.bigram_counts( corpus )

    # Get some statistics on the comparison of bigrams
    generated_corpus_bigrams  = compare_bigrams.bigram_counts ( corpus )
    generated_corpus_unigrams = compare_bigrams.unigram_counts( corpus )
    
    comp = compare_bigrams.compare_Ngrams( bigrams_unpermuted, generated_corpus_bigrams )
    comp = dict([ ("bigrams.%s"%k,v) for (k,v) in comp.items() ])
    comp["permutation"]=i+1
    
    unicomp = compare_bigrams.compare_Ngrams( unigrams_unpermuted, generated_corpus_unigrams )
    unicomp = dict([ ("unigrams.%s"%k,v) for (k,v) in unicomp.items() ])

    comp = {**comp,**unicomp} # merge the dicts
    comp["n.unique.bigrams"]  = len(generated_corpus_bigrams.keys())
    comp["n.unique.unigrams"] = len(generated_corpus_unigrams.keys())


    if True:

        print ("%i"%(i+1),end=" ",flush=True)

        # Quantify the copying
        cop = quantify_copying.corpus(corpus)

        corpus_stats = pd.concat([corpus_stats,
                                  pd.DataFrame({**comp,**cop},
                                               index=[i+1])])

        # Now also shuffle the word list, so that we don't always have the same lengths in the train and test
        # corpus after the following split.
        #random.shuffle(words)
        
        # Split into train and test corpus
        #train_corpus = words[:N_TRAIN]
        #test_corpus  = words[N_TRAIN:]

        # Write the shuffled corpora to file
        #f = open('%s/permutation_%05i_numbers.txt'%(OUTPUT_DIR,i+1),'w')
        #words_num = as_numeric.corpus_to_numeric( words )
        #corpus = "\n".join([ " ".join(w) for w in words_num ])
        #f.write(corpus)
        #f.close()


        f = open('%s/permutation_%05i.txt'%(OUTPUT_DIR,i+1),'w')
        corp = "\n".join([ " ".join(w) for w in corpus ])
        f.write(corp)
        f.close()




        #f = open('%s/permutation_%05i_test_corpus.txt'%(OUTPUT_DIR,i+1),'w')
        #corpus = "\n".join([ " ".join(w) for w in test_corpus ])
        #f.write(corpus)
        #f.close()

        i+=1

    else:
        pass # this permutation will be discarded

    n_tries +=1

corpus_stats.to_csv('interim/permutations_within_sentences_stats.csv')
print ("This took me %i tries."%n_tries)
