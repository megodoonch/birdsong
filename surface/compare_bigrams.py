

import numpy as np
import scipy.stats
import random
from bisect import bisect
import sys




"""
Ok, here the idea is that we compare the bigrams in each of the permutations
to those in the original corpus.
The bigrams form a discrete, unordered distribution. So we can compare them 
using the chi-squared statistic.
( see http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_HypothesisTesting-ChiSquare/BS704_HypothesisTesting-ChiSquare3.html )

Note that for chi-squared to work well, all cells should contain at least approx. 5 values.
(e.g. http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.chi2_contingency.html )
So I think we should remove cells that have less than 5 values.

Then the problem comes that we're no longer comparing chi^2 with the same number of observations.
It might be nice to have a way to correct for that.
Various measures of association exist between contingency tables, but most only work for the 2x2 case:
http://en.wikipedia.org/wiki/Contingency_table

http://www.csupomona.edu/~jlkorey/POWERMUTT/Topics/contingency_tables.html#assoc


Cramer's V seems like a nice solution. It works for larger than 2x2 cases and it is "normalised"
for the number of observations (always between 0 and 1).
Let's give it a try.

"""

def chisq_distribs( A, B ):
    # Given two distributions, calculate the chisq.
    
    assert len(A)==len(B)
    n = sum(A)+sum(B) # the total number of observations

    # The cardinality of the set of category labels
    #k=len(A)
    dat = np.matrix([A,B]) # they become rows in the matrix
    (r,k)=dat.shape


    # See e.g. http://stackoverflow.com/questions/19563126/chi-square-test-of-independence-in-python
    (chisq, p, df, expected_counts) = scipy.stats.chi2_contingency(dat)

    cramerV = np.sqrt( chisq/ (n*min([ k-1, r-1])) )

    return {"chisq"    :chisq,
            "n"        :n,
            "p"        :p,
            "df"       :df,
            "cramer.V" :cramerV
            }








def bigram_counts_from_flat_list( flat_words ):
    # Return a list of bigrams from a flat list of items
    # i.e. (word1, word2, word3, word4, word5, etc.)
    # There could be word boundaries in there, but they are identified
    # with their own proper symbol.
    bigram_counts = {}
    for i in range(len(flat_words)-1):
        
        bigr = ".".join(flat_words[i:(i+2)])
        #print bigr
        bigram_counts[bigr] = bigram_counts.get(bigr,0)+1
    return bigram_counts






def bigram_counts( words ):
    # Returns the bigram distributions of a list of sentences (i.e. list of list of words).

    # First turn into a flat list with word boundaries
    flat_words = ["#"]
    for w in words:
        flat_words+= w+["#"]

    # Return bigram counts
    return bigram_counts_from_flat_list(flat_words)







def bigram_counts_asymmetric( words ):
    # Returns the bigram distributions of a list of sentences (i.e. list of list of words).
    # Same as bigram_counts but this time we distinguish between start and end word boundary symbols.

    # First turn into a flat list with word boundaries
    flat_words = []
    for w in words:
        flat_words+= [">"]+w+["<"]

    # Return bigram counts
    bigrs = bigram_counts_from_flat_list(flat_words)
    bigrs = dict([ (b,cnt) for (b,cnt) in bigrs.items() if b!="<.>" ]) # ignore the stop-start transition
    
    return bigrs







def unigram_counts( sentences ):
    # Returns the bigram distributions of a list of sentences (i.e. list of list of words).
    # Simply ignores the word boundaries - who cares anyway.

    # First turn into a flat list with word boundaries
    unigrams = {}
    for s in sentences:
        for w in s:
            unigrams[w]=unigrams.get(w,0)+1
    # Return the counts
    return unigrams










def bigram_counts_noboundaries( sentences ):
    # Returns the bigram distributions of a list of sentences (i.e. list of list of words),
    # but it simply omits bigrams that involve a word boundary. Equivalently, we calculate
    # all word-internal bigrams for all words separately and merge those.

    bigrams = {}
    # First turn into a flat list with word boundaries
    for sentence in sentences:
        bigr = bigram_counts_from_flat_list(sentence)

        # Update our existing probabilities using these new bigram counts.
        for bi in bigr:
            bigrams[bi] = bigrams.get(bi,0)+bigr[bi]

    return bigrams









def compare_Ngrams( ngramsA, ngramsB ):
    # All right, given two dicts of bigrams with associated counts,
    # compare them.

    # All we have to do is generate a common set of bigrams
    all_ngrams = list(set(list(ngramsA.keys())+list(ngramsB.keys())))

    # ... and then evaluate each according to this common set,
    # substituting zero if the bigram in question does not occur
    countsA = list(map(lambda x: ngramsA.get(x,0),all_ngrams))
    countsB = list(map(lambda x: ngramsB.get(x,0),all_ngrams))

    return chisq_distribs(countsA,countsB)













def weighted_choice(choices):
    # Chooses from a list of (item,weight) where weight represents the weight
    # factor of that choice (the higher, the more likely it will get chosen).
    # Copied shamelessly from:
    # http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]






def generate_words_from_bigrams( bigram_counts, N_SENTENCES, maxlength, progress_output=True ):
    # Ok, this is admittedly a funny little function. What it does is it starts with a word
    # boundary (#) and goes on to generate bigrams in sequence, taking the probabilities
    # from bigram_counts. When it has finished a word, it goes into a list.
    # Like that, it keeps going until it has collected N_SENTENCES sentences.
    # Also, we don't generate any sentences longer than a particular length.
    # (that's just a little heuristic that shouldn't influence subsequent testing
    # because we will later choose sentences of particular lengths to match those
    # of the real corpus).
    # The results are presented in a dict where the keys are the sentence lengths
    # and the values are a list of (sentence,count) pairs, where count is the number
    # of times that particular sentence is generated (that, in turn, can be used
    # to generate sentences with veridical probabilities).

    n_generated = 0
    sentences = {} # this will hold the newborn sentences, ordered by their length

    last_symbol = "#"
    sentence = []

    transitions = {}
    # First, let's compute a transition matrix
    for k,count in bigram_counts.items():

        (prev,nxt) = k.split(".") # split the bigram in two
        transitions[prev] = transitions.get(prev,[])+[ (nxt,count) ]
        # So transitions[last_symbol] is a list of (next_symbol,count) 
        # values where $next_symbol has been observed $count times.

    
    last_percentile = -1

    while n_generated<N_SENTENCES:
        
        # Find the possible completions of the last_symbol
        next_bigrams = transitions[last_symbol]

        # Choose a next bigram by weighed choice using the observed frequencies in the transition matrix.
        last_symbol = weighted_choice(next_bigrams)

        if last_symbol=="#":
            # A new sentence is born!
            l = len(sentence)
            
            sentence_flat = ".".join(sentence)

            # If this sentence length didn't exist yet
            if l not in sentences:
                sentences[l] = {}
                
            thisl = sentences[l]
                
            # If this sentence didn't exist yet
            thisl[sentence_flat] = thisl.get(sentence_flat,0)+1
            
            sentence = [] # re-initialise
            n_generated+=1

            percentile = 100*n_generated/N_SENTENCES
            if progress_output:
                if percentile>last_percentile:
                    print ("%.0f%% "%percentile,end="",flush=True)
                    last_percentile = percentile
                    sys.stdout.flush()
        else:
            # Otherwise keep adding to the existing sentence
            sentence+=[last_symbol]
            if len(sentence)>maxlength:
                # Reset! And don't add to the list of generated sentences because this is not a complete sentence.
                last_symbol="#"
                sentence=[]
        

    # Now slightly reorder the list so that it is easier to select items
    for l in sentences:
        sentences[l] = list(sentences[l].items() )
        # Instead of a dict, the values now are (sentence,count)

    return sentences
        

    









if False:

    # Example usage of the chisq distribution testing
    distribA = [1, 2, 5, 8]
    distribB = [1, 4, 10, 16]
    print (chisq_distribs(distribA,distribB))


