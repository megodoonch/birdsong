"""
expectation maximisation algorithm for two-part grammar

"""


import numpy as np
import random
import copy

import markhov
import fsa
import bigrams

from markhov import log0, log_add, log

ops=fsa.ops
bigrams=bigrams.bigrams

f=open('../corpus/cath8.txt','r')
corpus = f.readlines()
f.close()
corpus = [line.rstrip('\n') for line in corpus]


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

def bi_count(bi,bis):
    (a,b)=bi
    return len([i for i in range(len(bis)-1) if bis[i]==a and bis[i+1]==b])

def uni_count(word,bis):
    return len([w for w in bis if w==word] )

def bi_counts(bis):
    ts={}
    for i in range(1,len(bis)):
        a,b=bis[i-1],bis[i]
        if (a,b) not in ts: # only calculate this once
            bc = bi_count((a,b),bis)
            ts.append(((q1,e,q2),log(tc)))
    return ts


def scs(route):
    ss={}
    (qs,es)=route
    for state in qs:
        if state not in ss:
            sc=state_count(route,state)
            ss[state]=log(sc)
    return ss


def tcs(route):
    """

    Returns
    dict of (transition):log count
    """
    (qs,es)=route
    ts={}
    for i in range(1,len(qs)):
        q1,e,q2=qs[i-1],es[i-1],qs[i]
        if (q1,e,q2) not in ts:
            tc = transition_count(route,(q1,e,q2))
            ts[(q1,e,q2)]=log(tc)
    return ts



# def update_t(t,s,bigrams,fsa,start='S'):
#     parses = parse(s,bigrams,fsa,start)
#     tot=log0
#     p_s=log0
#     for (bis,_,route,__,p) in parses:
#         tot = log_add(tot, p+transition_count(route,t)-state_count(route,t[0]))
#         p_s = log_add(p_s,p)
#    return tot-p_s


# def update_t_first(t,corpus,bigrams,fsa,start='S',end='F'):
#     """
#     gets a new probability for a transition based on a corpus
#     We get this by summing over all the times we use the transition (x p(parse | sent))
#     and dividing that by all the times we were in the origin state of that transition
#     (x p(parse | sent))

#     Arguments
#     t      : transition (origin state, emission, desitination state)
#     corpus : string list
#     bigrams: bigram markhov chain
#     fsa    : operations FSA
#     start  : start state

#     """
#     tot_tc = log0
#     tot_sc = log0
#     for s in corpus:
#         #print (s)
#         #print (tot_tc)
#         parses = parse(s,bigrams,fsa,start) # parse the sentence
#         parses = clean_parses(parses) # just keep (bis,route,p)
#         p_s = p_sent(parses,bigrams,fsa,start,end) # get the sent prob
#         for (bis,route,p) in parses:
#             # add in p(parse | sent) * TC 
#             tot_tc = log_add(tot_tc, p-p_s+log(transition_count(route,t)))
#             # add in p(parse | sent) * SC
#             tot_sc = log_add(tot_sc, p-p_s+log(state_count(route,t[0])))
            
#     #print (tot_tc,tot_sc)
#     return tot_tc-tot_sc # counts(transition)/counts(state)



def parse_corpus(corpus,bigrams,fsa,start='S',end='F'):
    """
    parse the sentences and create a corpus (s,parses,prob s)
    """
    sents = []
    for s in corpus:
        parses = markhov.parse(s,bigrams,fsa,start) # parse the sentence
        parses = markhov.clean_parses(parses) # just keep (bis,route,p)
        p_s=log0
        parses_with_counts = []
        for (big,route,p) in parses:
            t_counts=tcs(route) # calculate transition counts
            s_counts=scs(route) # calculate state counts
            parses_with_counts.append((big,route,p,s_counts,t_counts))
            p_s=log_add(p_s,p) # add to sentence prob

        sents.append((s,parses_with_counts,p_s))
    return sents


       
def update_fsa(corpus,bigrams,fsa):
    """
    updates the operations fsa
    
    Arguments
    corpus : a parsed corpus of triples (s,parses,prob)


    """
    new_fsa = copy.deepcopy(fsa)
    for lhs in new_fsa:
        #print (lhs)
        for rhs in new_fsa[lhs]:
            #print (rhs)
            for e in new_fsa[lhs][rhs]:
                #print (e)
                tot_tc = log0
                tot_sc = log0
                
                for (s,parses,p_s) in corpus:
                    #print (s)
                    for (bis,route,p,tc,sc) in parses:
                        #print (tc)
                        #print (sc)
                        if (lhs,e,rhs) in tc:
                            # add in p(parse | sent) * TC 
                            tot_tc = log_add(tot_tc, p-p_s+tc[(lhs,e,rhs)])
                        if lhs in sc:
                            # add in p(parse | sent) * SC
                            tot_sc = log_add(tot_sc, p-p_s+sc[lhs])
            new_fsa[lhs][rhs][e]=tot_tc-tot_sc # update prob
    return new_fsa
