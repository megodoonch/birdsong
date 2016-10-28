

""" Here we generate corpora using bigram probabilities of the actually observed sentences,
but we do not impose any length restrictions on the words: we just generate them as they come.
This might tell us whether the amount of copying previously observed can be explained by length
restrictions. """



import random
import pandas as pd
import compare_bigrams
import sys
import quantify_copying
import multiprocessing as mp
import os


# The file that contains the base corpus
INPUT_FILE = "../corpus/cath8.txt"
# Each line should be a sentence, with the words separated by spaces

OUTPUT_DIR = "./output/bigramgen-free/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# The number of sentences to generate in our super corpus
#N_SENTENCES = 500000

# How many corpora to generate. Each corpus will have the same
# number of sentences of the same lengths. Also, we hope that we 
# will be able to find similar unigram and bigram distributions.
# Let's see.
N_CORPORA = 1000



# The size of the train corpus (in number of sentences)
#N_TRAIN = 184


# Read the input file and obtain a list of list of strings, i.e. the list of sentences
f = open(INPUT_FILE,'r')
lines = f.readlines()
f.close()
cath8 = [ l.strip().split(" ") for l in lines ]



N_WORDS_IN_CORPUS = sum([ len(s) for s in cath8 ])

maxlength = 999999 # just don't put any realistic limit


# Get a list of the bigrams, but omitting those involving word boundaries.
bigrams_unpermuted  = compare_bigrams.bigram_counts( cath8 )
unigrams_unpermuted = compare_bigrams.unigram_counts( cath8 )





def generate_bigram_corpus(i):

    print (i,end=" ",flush=True)

    # First make a candidate corpus
    corpus = []
    total_length = 0
    while total_length<N_WORDS_IN_CORPUS: # we obtain a sentence of a particular length

        # Generate one item
        sent,_ = list(compare_bigrams.generate_words_from_bigrams(bigrams_unpermuted, 1, maxlength, progress_output=False).values())[0][0]
        sent = sent.split(".")
        corpus+=[sent]
        total_length+= len(sent)
        


    # Ok, so now we have a corpus that is just slightly bigger in number of words than the target corpus.
    # We consider discarding the last sentence, and seeing if we end up closer to the target length.
    dlength = abs(total_length - N_WORDS_IN_CORPUS)
    dlength_discard = abs((total_length-len(corpus[-1]))-N_WORDS_IN_CORPUS) # the length of the corpus if we would discard the final addition

    if dlength_discard<dlength: # if, discarding the last sentence, we are closer to the target length, do the discard
        corpus = corpus[:-1]

    total_length = sum([len(s) for s in corpus])


    generated_corpus_bigrams  = compare_bigrams.bigram_counts ( corpus )
    generated_corpus_unigrams = compare_bigrams.unigram_counts( corpus )
    
    comp = compare_bigrams.compare_Ngrams( bigrams_unpermuted, generated_corpus_bigrams )
    comp = dict([ ("bigrams.%s"%k,v) for (k,v) in comp.items() ])
    
    unicomp = compare_bigrams.compare_Ngrams( unigrams_unpermuted, generated_corpus_unigrams )
    unicomp = dict([ ("unigrams.%s"%k,v) for (k,v) in unicomp.items() ])

    # Also quantify copying
    cop = quantify_copying.corpus(corpus)
    
    # Combine all metrics we've gathered about this corpus
    comp = {**comp,**unicomp,**cop} # dict(comp.items()+unicomp.items()+cop.items()) # merge the dicts
    comp["n.unique.bigrams"]  = len(generated_corpus_bigrams.keys())
    comp["n.unique.unigrams"] = len(generated_corpus_unigrams.keys())


    if True:

        comp["randomisation"]=i
        corpus_stats = pd.DataFrame(comp,index=[i])
    
        # Write the shuffled corpora to file

        # Now also shuffle the word list, so that we don't always have the same lengths in the train and test
        # corpus after the following split.
        #random.shuffle(corpus)
        
        # Split into train and test corpus
        #train_corpus = corpus[:N_TRAIN]
        #test_corpus  = corpus[N_TRAIN:]

        # Write the shuffled corpora to file
        f = open('%s/bigramfree_%05i_corpus.txt'%(OUTPUT_DIR,i+1),'w')
        corpus = "\n".join([ " ".join(w) for w in corpus ])
        f.write(corpus)
        f.close()


        #f = open('%s/bigramfree_%05i_test_corpus.txt'%(OUTPUT_DIR,i+1),'w')
        #corpus = "\n".join([ " ".join(w) for w in test_corpus ])
        #f.write(corpus)
        #f.close()

        return corpus_stats
    return None







corpus_stats = pd.DataFrame()


if True:
    pool = mp.Pool(processes=6)
    results = [pool.apply_async(generate_bigram_corpus, args=(i,)) for i in range(N_CORPORA)]
    output = [p.get() for p in results]

corpus_stats = pd.concat(output)







    

corpus_stats.to_csv('interim/bigramgen_free_corpus_stats.csv')



