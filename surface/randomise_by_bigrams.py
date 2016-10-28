
# This is a novel approach to generating a random corpus.
# Instead of making it completely random we generate samples
# from the bigram distribution. This will ensure that we have a relatively
# good bigram distribution and possibly also unigram distribution.
# We'll check for both using our goodness-of-fit parameters as before.


import random
import pandas as pd
import compare_bigrams
import sys
import quantify_copying
import os
import multiprocessing as mp


# The file that contains the base corpus
INPUT_FILE = "../corpus/cath8.txt"
# Each line should be a sentence, with the words separated by spaces

OUTPUT_DIR = "./output/bigramgen/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# The number of sentences to generate in our super corpus
N_SUPERCORPUS_SENTENCES = 1000000

# How many corpora to generate. Each corpus will have the same
# number of sentences of the same lengths. Also, we hope that we 
# will be able to find similar unigram and bigram distributions.
# Let's see.
N_CORPORA = 1000


# Any permutations that have Cramer-V below this limit will
# be accepted, the rest will be discarded.
# Set this to 1.0 to not discard anything.
MAX_UNIGRAM_CRAMER_V = 1.0
MAX_BIGRAM_CRAMER_V = 1.0



# Read the input file and obtain a list of list of strings, i.e. the list of sentences
f = open(INPUT_FILE,'r')
lines = f.readlines()
f.close()
sentences = [ l.strip().split(" ") for l in lines ]



# Get a list of the bigrams, but omitting those involving word boundaries.
bigrams_unpermuted  = compare_bigrams.bigram_counts( sentences )
unigrams_unpermuted = compare_bigrams.unigram_counts( sentences )


# Determine the length of each sentence
lengths = [ len(s) for s in sentences ]
maxlength = max(lengths) # the maximum sentence length

print ("Generating the super corpus...")

# Generate a whole huge lot of sentences from them, storing them all in a
# gigantuous dict, where the keys are the sentence lengths.
supercorpus = compare_bigrams.generate_words_from_bigrams(
    bigrams_unpermuted,
    N_SUPERCORPUS_SENTENCES,
    maxlength)

print ("...done")





def generate_bigram_corpus(i):

    # This generates one bigram corpus (randomisation i)
    # outputs it to a file, and returns the corpus stats.

    print (i,)
    sys.stdout.flush()
    
    # First make a candidate corpus
    corpus = []
    for l in lengths: # we obtain a sentence of a particular length

        # Choose a sentence of that length from the randomly generated sentences
        #sent = random.choice(supercorpus[l])
        sent = compare_bigrams.weighted_choice(supercorpus[l])
        # Sent will be encoded as a.b.c.d, so now we turn it back into a list, and add it to the corpus:
        corpus.append(sent.split("."))


    # So now we should have a corpus of the same number of words,
    # the same distribution of word lengths as those of the original corpus.
    # What we need to check is whether the distribution of unigrams and bigrams
    # is approximately correct. Let's see.
    # As a metric we use the Cramer V metric.

    generated_corpus_bigrams  = compare_bigrams.bigram_counts ( corpus )
    generated_corpus_unigrams = compare_bigrams.unigram_counts( corpus )
    
    comp = compare_bigrams.compare_Ngrams( bigrams_unpermuted, generated_corpus_bigrams )
    comp = dict([ ("bigrams.%s"%k,v) for (k,v) in comp.items() ])

    unicomp = compare_bigrams.compare_Ngrams( unigrams_unpermuted, generated_corpus_unigrams )
    unicomp = dict([ ("unigrams.%s"%k,v) for (k,v) in unicomp.items() ])

    comp["n.unique.bigrams"]  = len(generated_corpus_bigrams.keys())
    comp["n.unique.unigrams"] = len(generated_corpus_unigrams.keys())

    # Also quantify copying
    cop = quantify_copying.corpus(corpus)
    
    # Combine all metrics we've gathered about this corpus
    comp = {**comp,**unicomp,**cop} # merge the dicts
    

    # Hope this happens reasonably often
    if True:

        comp["permutation"]=i

        stats = pd.DataFrame(comp,index=[i])
    
        # Write the shuffled corpora to file

        # Now also shuffle the word list, so that we don't always have the same lengths in the train and test
        # corpus after the following split.
        #random.shuffle(corpus)
        
        # Split into train and test corpus
        #train_corpus = corpus[:N_TRAIN]
        #test_corpus  = corpus[N_TRAIN:]

        # Write the shuffled corpora to file
        f = open('%s/permutation_%05i.txt'%(OUTPUT_DIR,i),'w')
        corpus = "\n".join([ " ".join(w) for w in corpus ])
        f.write(corpus)
        f.close()


        #f = open('%s/bigramgen_%05i_test_corpus.txt'%(OUTPUT_DIR,i+1),'w')
        #corpus = "\n".join([ " ".join(w) for w in test_corpus ])
        #f.write(corpus)
        #f.close()


        return stats

    return None






if True:
    pool = mp.Pool(processes=6)
    results = [pool.apply_async(generate_bigram_corpus, args=(i+1,)) for i in range(N_CORPORA)]
    output = [p.get() for p in results]

corpus_stats = pd.concat(output)


corpus_stats.to_csv('interim/bigramgen_corpus_stats.csv')



