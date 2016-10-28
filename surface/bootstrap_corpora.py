


import pandas as pd
import random
import compare_bigrams
import quantify_copying

import sys



if True:
    # Let's have a look. If we randomly take (with replacement) from the corpus
    # to generate a new corpus with an equal number of words (i.e. bootstrap),
    # what kind of unigram and bigram distributions do we end up with and how
    # far or close are they from what we end up with using permutation-based or randomised
    # corpora?



    N_BOOTSTRAP_SAMPLES = 1000

    # The file that contains the base corpus
    INPUT_FILE = "../corpus/cath8.txt"
    # Each line should be a sentence, with the words separated by spaces


    # Read the input file and obtain a list of list of strings, i.e. the list of sentences
    f = open(INPUT_FILE,'r')
    lines = f.readlines()
    f.close()
    sentences = [ l.strip().split(" ") for l in lines ]


    corpus_stats = pd.DataFrame()



    # Get a list of the bigrams, but omitting those involving word boundaries.
    bigrams_unpermuted  = compare_bigrams.bigram_counts( sentences )
    unigrams_unpermuted = compare_bigrams.unigram_counts( sentences )


    for i in range(N_BOOTSTRAP_SAMPLES):

        print (i,end=" ",flush=True)

        # Take a bootstrap sample
        bootstrap_corpus = []
        for j in range(len(sentences)):
            bootstrap_corpus.append( random.choice(sentences) )


        # So now we have a bootstrap resampled corpus with the same number
        # of sentences but not necessarily of the same lengths, nor necessarily
        # with the same unigram or bigram distributions.
        generated_corpus_bigrams  = compare_bigrams.bigram_counts ( bootstrap_corpus )
        generated_corpus_unigrams = compare_bigrams.unigram_counts( bootstrap_corpus )

        comp = compare_bigrams.compare_Ngrams( bigrams_unpermuted, generated_corpus_bigrams )
        comp = dict([ ("bigrams.%s"%k,v) for (k,v) in comp.items() ])
        comp["permutation"]=i

        unicomp = compare_bigrams.compare_Ngrams( unigrams_unpermuted, generated_corpus_unigrams )
        unicomp = dict([ ("unigrams.%s"%k,v) for (k,v) in unicomp.items() ])
        comp = {**comp,**unicomp} # merge the dicts
        comp["n.unique.bigrams"]  = len(generated_corpus_bigrams.keys())
        comp["n.unique.unigrams"] = len(generated_corpus_unigrams.keys())

        # Quantify the copying
        cop = quantify_copying.corpus(bootstrap_corpus)

        corpus_stats = corpus_stats.append(pd.DataFrame({**comp,**cop},index=[i]))



    corpus_stats.to_csv('interim/bootstrap_corpora_stats.csv')
