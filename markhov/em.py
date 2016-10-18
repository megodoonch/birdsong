"""
expectation maximisation algorithm for two-part grammar

"""


import numpy as np
import random
import copy

import markhov
import fsa
import bigrams

ops=fsa.ops
bigrams=bigrams.bigrams


def state_count(route,state):
    """given a route and a particular state, returns the number of times that state occurs

    Arguments
    route : (qs,es)
    state : state from ops FSA
    """
    (qs,_)=route
    return len([q for q in qs if q==state])

def transition_count(route,trans):
    """given a route and a particular transition,
        returns the number of times that transition occurs

    Arguments
    route : (qs,es)
    transition : (q1,e,q2) where q1,q2 are states and e is an operation
    """
 
    (qs,es)=route
    (q1,e,q2)=trans
    return len([i for i in range(len(qs)-1) if qs[i]==q1 and qs[i+1]==q2 and es[i]==e])

# def update_t(t,s,bigrams,fsa,start='S'):
#     parses = parse(s,bigrams,fsa,start)
#     tot=log0
#     p_s=log0
#     for (bis,_,route,__,p) in parses:
#         tot = log_add(tot, p+transition_count(route,t)-state_count(route,t[0]))
#         p_s = log_add(p_s,p)
#    return tot-p_s


def update_t_first(t,corpus,bigrams,fsa,start='S',end='F'):
    """
    gets a new probability for a transition based on a corpus
    We get this by summing over all the times we use the transition (x p(parse | sent))
    and dividing that by all the times we were in the origin state of that transition
    (x p(parse | sent))

    Arguments
    t      : transition (origin state, emission, desitination state)
    corpus : string list
    bigrams: bigram markhov chain
    fsa    : operations FSA
    start  : start state

    """
    tot_tc = log0
    tot_sc = log0
    for s in corpus:
        #print (s)
        #print (tot_tc)
        parses = parse(s,bigrams,fsa,start) # parse the sentence
        parses = clean_parses(parses) # just keep (bis,route,p)
        p_s = p_sent(parses,bigrams,fsa,start,end) # get the sent prob
        for (bis,route,p) in parses:
            # add in p(parse | sent) * TC 
            tot_tc = log_add(tot_tc, p-p_s+log(transition_count(route,t)))
            # add in p(parse | sent) * SC
            tot_sc = log_add(tot_sc, p-p_s+log(state_count(route,t[0])))
            
    #print (tot_tc,tot_sc)
    return tot_tc-tot_sc # counts(transition)/counts(state)



       
def update_fsa_first(corpus,bigrams,fsa,start='S',end='F'):
    """
    updates the operations fsa on the first iteration

    """
    new_fsa = copy.deepcopy(fsa)
    sents = []
    for s in corpus:
        parses = parse(s,bigrams,fsa,start) # parse the sentence
        parses = clean_parses(parses) # just keep (bis,route,p)
        p_s = p_sent(parses,bigrams,fsa,start,end) # get the sent prob
        sents.append((s,parses,p_s))
    for lhs in fsa:
        for rhs in fsa[lhs]:
            for e in fsa[lhs][rhs]:
                tot_tc = log0
                tot_sc = log0
                
                for (s,parses,p_s) in sents:
                    for (bis,route,p) in parses:
                        # add in p(parse | sent) * TC 
                        tot_tc = log_add(tot_tc, p-p_s+log(transition_count(route,(lhs,e,rhs))))
                        # add in p(parse | sent) * SC
                        tot_sc = log_add(tot_sc, p-p_s+log(state_count(route,lhs)))
            new_fsa[lhs][rhs][e]=tot_tc-tot_sc
    return new_fsa
