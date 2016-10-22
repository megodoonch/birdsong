"""turning corpus into bigram chain"""

import numpy as np
import random


f=open('../corpus/cath8.txt','r')
corpus = f.readlines()
f.close()
corpus = [line.rstrip('\n') for line in corpus]



bis = {'[':{}}
for s in corpus:
    s=['[']+s.split(' ')
    for i in range(1,len(s)):
        bis[s[i-1]]=bis.get(s[i-1],{})
        bis[s[i-1]][s[i]]=bis[s[i-1]].get(s[i],0)+1

for lhs in bis:
    tot=float(sum(bis[lhs].values()))
    for rhs in bis[lhs]:
        bis[lhs][rhs]=bis[lhs][rhs]/tot


def bis_log(bigrams):
    for a in bigrams:
        for b in bigrams[a]:
            bigrams[a][b]=np.log(bigrams[a][b])
    return bigrams

bigrams = bis_log(bis)

def bis_random(bigrams):
    """
    gives bigrams random probabilities that sum to 1 for each lhs

    Arguments
    bigrams : bigram markhov chain as a dict of dicts

    Returns
    bigrams with probs replaced with random (log) probs
    """
    for a in bigrams:
        n=len(bigrams[a])
        print (n)
        r = [random.random() for i in range(n)]
        s = sum(r)
        r = [ i/s for i in r ]
        i=0
        print (r[i])
        for b in bigrams[a]:
            bigrams[a][b] = np.log(r[i])
            i+=1
    return bigrams
