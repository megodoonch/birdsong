"""
expectation maximisation algorithm for two-part grammar

"""


import numpy as np
import random
import copy

import markhov
import fsa
import bigrams

from toy_corpus import *

from markhov import log0, log_add, log, log_sum

ops=fsa.ops
bigrams=bigrams.bigrams
bi_ops=fsa.bi_ops

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
    """
    gets the number of occurances of bigram in bigrams

    Arguments
    bi  : bigram (a,b)
    bis : sequence of alphabet members 

    Returns
    int

    """
    (a,b)=bi
    return len([i for i in range(len(bis)-1) if bis[i]==a and bis[i+1]==b])

def uni_count(word,bis):
    """
    gets the number of occurances of word in bigrams

    Arguments
    word : alphabet member
    bis  : sequence of alphabet members. We don't count the last one because we never left it

    Returns
    int

    """

    return len([w for w in bis[:-1] if w==word] )

def bi_counts(bis):
    """
    gets the number of occurances of all bigrams in sequence

    Arguments
    bis : sequence of alphabet members 

    Returns
    dict of (a,b):count

    """

    ts={}
    for i in range(1,len(bis)):
        a,b=bis[i-1],bis[i]
        if (a,b) not in ts: # only calculate this once
            bc = bi_count((a,b),bis)
            ts[(a,b)]=bc
    return ts

def uni_counts(bis):
    """
    gets the number of occurances of all words in sequence

    Arguments
    bis : sequence of alphabet members. We don't count the last one because we never left it

    Returns
    dict of wd:count

    """
    us={}
    for a in bis[:-1]:
        if a not in us: # only calculate this once
            uc = uni_count(a,bis)
            us[a]=uc
    return us


def scs(route):
    """
    gets the number of occurances of all states in route (Q,E)

    Arguments
    route : pair (states, emissions) of a route through the operations FSA

    Returns
    dict of q:count

    """
    ss={}
    (qs,es)=route
    for state in qs:
        if state not in ss:
            sc=state_count(route,state)
            ss[state]=sc
    return ss


def tcs(route):
    """
    gets the number of occurances of all transitions in route (Q,E)

    Arguments
    route : pair (states, emissions) of a route through the operations FSA

    Returns
    dict of (transition): count
    """
    (qs,es)=route
    ts={}
    for i in range(1,len(qs)):
        q1,e,q2=qs[i-1],es[i-1],qs[i]
        if (q1,e,q2) not in ts: # only do this once
            tc = transition_count(route,(q1,e,q2))
            ts[(q1,e,q2)]=tc
    return ts



def parse_corpus(corpus,bigrams,fsa,start='S',end='F'):
    """
    parse the sentences and create a corpus (s,parses,prob s)

    Arguments
    corpus  : string list
    bigrams : bigram morkhov chain (dict)
    fsa     : operations FSA (dict)

    Returns
    a parsed corpus of triples (s,parses,prob)
       where parses is a list of
       (bis,route,prob,state counts, trans counts,
         bigram counts, unigram counts)

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
            bigram_counts = bi_counts(big)
            unigrams = uni_counts(big)
            parses_with_counts.append((big,route,p,s_counts,t_counts,bigram_counts,unigrams))
            p_s=log_add(p_s,p) # add to sentence prob

        sents.append((s,parses_with_counts,p_s))
    return sents





def p_corpus(parsed_corpus):
    """
    Computes the log likelihoods of the corpus given the latest grammar
    These are already computed and stored in the parsed corpus

    """
    return sum([p for (s,parses,p) in parsed_corpus])


def sum_p_corpus(parsed_corpus):
    return  log_sum([p for (s,parses,p) in parsed_corpus])




def update_automata(corpus,bigrams,fsa,verbose=False):
    """
    updates the bigram chain and operations fsa
    
    Arguments
    corpus : a parsed corpus of triples (s,parses,prob)
              where parses is a list of
              (bis,route,prob,state counts, trans counts,
              bigram counts, unigram counts)
    bigrams : bigram morkhov chain (dict)
    fsa     : operations FSA (dict)

    returns
    updated bigrams, updated fsa

    """

    #TODO smoothing for unused rules
    new_fsa = copy.deepcopy(fsa) #copy the ops FSA
    for lhs in new_fsa: # go through the FSA
        if verbose: print (lhs)
        for rhs in new_fsa[lhs]:
            #print (rhs)
            for e in new_fsa[lhs][rhs]:
                #print (e)
                #print (lhs,e,rhs)
                tot_tc=log0
                tot_sc=log0

                for (s,parses,p_s) in corpus: # go through parsed corpus
                    #print (s)
                    # add up all the TCs/SCs for the parses of this sentence
                    for (bis,route,p,sc,tc,bc,uc) in parses:
                        #print (tc)
                        #print (sc)
                        if (lhs,e,rhs) in tc: # if this parse has this rule
                            # add in counts, times p(parse)
                            tot_tc=log_add(tot_tc, p-p_s + log(tc[(lhs,e,rhs)]))
                            #print ("tot_tc: %f"%tot_tc)
                        if lhs in sc:
                            tot_sc=log_add(tot_sc, p-p_s + log(sc[lhs]))
                            #print ("tot_sc: %f"%tot_sc)

                        # when you've got all the parses for this sentence processed, 
                        #divide by the prob of the sentence 
                        #and add the result into the new prob for the rule
                        #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
                        #print (lhs,e,rhs,new_prob)
                # when you're through the corpus, update prob for this rule

                if verbose: print (" new prob for %s %s %s: %f - %f = %f"%(lhs,e,rhs,tot_tc,tot_sc,tot_tc-tot_sc))
                if tot_sc!=log0: # only make a new prob if we used this state                    
                    new_fsa[lhs][rhs][e]= tot_tc-tot_sc
                else:
                    n_rhs = len(fsa[lhs])
                    new_fsa[lhs][rhs][e]=0.-log(n_rhs) # 1/n_rhs for this state
                    if verbose: print (" %s not used; use %.2f instead"%(lhs,0.-log(n_rhs)))
                    #new_fsa[lhs][rhs][e]=fsa[lhs][rhs][e] # keep old p
                #print (new_fsa[lhs][rhs][e])
    #print (new_fsa)

    if verbose: print ("\n Bigrams")
    new_bis = copy.deepcopy(bigrams) # copy bigrams
    for a in bigrams: #go through the chain
        if verbose: print (a)
        for b in bigrams[a]:
            #print (a,b)
            tot_bc=log0
            tot_uc=log0
            for (s,parses,p_s) in corpus: # go through the corpus
                #new_p_this_s=log0
                for (bis,route,p,sc,tc,bc,uc) in parses:
                    if (a,b) in bc: # look for bigram in counts
                        # add in counts, times p(parse)
                        tot_bc=log_add(tot_bc, p-p_s + log(bc[(a,b)]))
                    if a in uc:
                        tot_uc=log_add(tot_uc, p-p_s + log(uc[a]))
                #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
            
            if tot_uc!=log0: # only make a new prob if we used this rule
                if verbose: print (" new prob for %s %s: %f - %f = %f"%(a,b,tot_bc,tot_uc,tot_bc-tot_uc))
                new_bis[a][b]=tot_bc-tot_uc # new prob
            else: 
                n_rhs = len(bigrams[a])
                new_bis[a][b]=0.-log(n_rhs)
                if verbose: print (" %s not used; use %.2f instead"%(a,0.-log(n_rhs)))
                #new_bis[a][b]=bigrams[a][b] # keep old prob
            
    return new_bis,new_fsa



def check_fsa(fsa):
    ok=True
    for lhs in fsa:
        if len(fsa[lhs])>0:
            tot=log0
            for rhs in fsa[lhs]:
                for e in fsa[lhs][rhs]:
                    tot=log_add(tot,fsa[lhs][rhs][e])
            if not np.isclose(tot,0.):
                print (lhs)
                ok=False
    return ok

def check_bis(bigrams):
    ok=True
    for a in bigrams:
        if len(bigrams[a])>0:
            tot=log0
            for b in bigrams[a]:
                tot=log_add(tot,bigrams[a][b])
            if not np.isclose(tot,0.):
                print (a)
                ok=False
    return ok



def em_main(parsed_corpus,bigrams,fsm,n,start='S',end='F',verbose=False):
    """
    iterates expectation maximisation n times

    Arguments
    parsed_corpus = list of triples (s,parses,prob)
              where parses is a list of
              (bis,route,prob,state counts, trans counts,
              bigram counts, unigram counts)
    bigrams : bigram morkhov chain (dict)
    fsm     : operations FSA (dict)
    n       : number of times to run EM
    start   : start state in ops
    end     : final state in ops

    returns
    updated bigrams, updated fsm

    """
    def inner(pc,bigrams,fsm):
        """
        updates the automata
         and then uses the new automata to update the probs in the corpus

        Arguments
        pc     : parsed corpus
        bigrams: bigram chain
        fsm    : ops FSM

        Returns
        updated corpus,bigrams,fsm
         
        """
        (new_bis,new_fsa)=update_automata(parsed_corpus,bigrams,fsm)
        #print (new_fsa)
        #update the corpus
        new_c=[]
        for (s,parses,p_s) in parsed_corpus: 
            #print (s)
            p_s=log0
            new_parses=[]
            for (bis,route,p,sc,tc,bc,uc) in parses:
                #calculate the new prob of the parse using the new grammar
                p=markhov.p_parse((bis,route,p,sc,tc,bc,uc),
                                new_bis,new_fsa,start,end)
                # replace the old p with the new one 
                #and put this parse into the list of parses
                new_parses.append((bis,route,p,sc,tc,bc,uc))
                p_s=log_add(p_s,p) # add in this p to the total prob of the sentence

            new_c.append((s,new_parses,p_s)) # here's our new parsed sentence
        return new_c,new_bis,new_fsa

    corpora = [parsed_corpus]
    bigram_chains = [bigrams] # keep all the versions of the grammar
    ops_fsas = [fsm] # keep all the versions of the grammar
    i=0 # counter for iterations
    if verbose: print ("initial LL: %.2f"%p_corpus(parsed_corpus)) # LL (corpus | grammar)
    if verbose: print ("iteration %i"%i)
    if n>0: # if we said 0 we'd better not do anything
        # first we use the given corpus and automata
        new_c,new_bis,new_fsa=inner(parsed_corpus,bigrams,fsm)
        corpora.append(new_c)
        bigram_chains.append(new_bis)
        ops_fsas.append(new_fsa)
        if verbose: print ("LL: %.2f"%p_corpus(new_c)) # LL (corpus | new grammar)

    i+=1
    while i<n:
        if verbose: print ("iteration %i"%i)
        # now we have new automata to iterate over
        new_c,new_bis,new_fsa=inner(new_c,new_bis,new_fsa)
        if not check_fsa(new_fsa):
            print ("Ops FSA invalid at iteration %i")
            print (fsa2string(new_fsa))
        corpora.append(new_c)
        bigram_chains.append(new_bis)
        ops_fsas.append(new_fsa)
        if verbose: print ("LL: %.2f"%p_corpus(new_c)) # LL (corpus | new grammar)
        i+=1

    print (fsa2string(ops_fsas[-1]))
    return corpora,bigram_chains,ops_fsas



def em(corpus,bigrams,fsm,n,start='S',end='F',verbose=False):
    """
    wrapper parses corpus and then iterates expectation maximisation n times

    Arguments
    corpus = list of strings
    bigrams : bigram morkhov chain (dict)
    fsm     : operations FSA (dict)
    n       : number of times to run EM
    start   : start state in ops
    end     : final state in ops

    returns
    updated bigrams, updated fsm

    """
    # parse the corpus and store the result
    print ("Parsing corpus...")
    parsed_corpus = parse_corpus(corpus,bigrams,fsm,start)
    print ("Running EM %i times"%n)
    # run EM n times
    return em_main(parsed_corpus,bigrams,fsm,n,start,end,verbose)



def em_train(train,test,bigram,fsm,n,start='S',end='F'):
    """
    Runs EM once, since that seems to be all we need, on training corpus
     and reports the log likelihoods of the training and testing corpora

    Arguments
    train  : 1/3 of the corpus
    test   : 2/3 of the corpus
    bigram : bigram markhov chain
    fsm    : operations FSM
    start  : start state in fsm
    end    : final state in fsm

    Returns
    log likelihood of the test corpus

    """

    # train
    new_corpora,new_bigrams,new_fsm = em(train,bigrams,fsm,n,start,end)
    print ("LL training corpus: %.2f"%p_corpus(new_corpora[1]))
    # parse the test corpus with the trained grammar
    parsed_test = parse_corpus(test,new_bigrams[1],new_fsm[1])
    ll = p_corpus(parsed_test)
    print ("LL test corpus: %.2f"%ll)
    return ll
        

def compare(train,test,bigrams,fsm_copy,fsm_no_copy,n,start='S',end='F',verbose=False):
    """
    Trains two different FSAs and the same bigrams, but seperately, on the training corpus.
    Tests on the test corpus

    Arguments
    train      : training corpus (part of the corpus)
    test       : testing corpus (the rest of the corpus)
    bigrams    : bigram markhov chain
    fsm_copy   : FSM with copy rules
    fsm_no_copy: FSM without copy rules
    start      : start state in FSMs
    end        : final state in FSMs

    Note that the two FSAs don't have to be copy/no copy, but that's the way they'll be referred to in printing

    Returns
    (log likelihood of test corpus given the copy grammar,
     log likelihood of test corpus given the no-copy grammar)
    """

    
    print ("\nCopy")
    # train copy grammar
    new_corpora_copy,new_bigrams_copy,new_fsm_copy = em(train,bigrams,fsm_copy,n,start,end)
    print ("Copy LL training corpus: %.2f"%p_corpus(new_corpora_copy[-1]))
    #parse the test corpus with the trained grammar
    parsed_test_copy = parse_corpus(test,new_bigrams_copy[-1],new_fsm_copy[-1])
    ll_copy = p_corpus(parsed_test_copy) # log likelihood (test | grammar)
    print ("Copy LL test corpus: %.2f"%ll_copy)
    
    print ("\nNo Copy")
    # train no-copy grammar
    new_corpora_no_copy,new_bigrams_no_copy,new_fsm_no_copy = em(train,bigrams,fsm_no_copy,n,start,end)
    print ("No Copy LL training corpus: %.2f"%p_corpus(new_corpora_no_copy[-1]))
    #parse the test corpus with the trained grammar
    parsed_test_no_copy = parse_corpus(test,new_bigrams_no_copy[-1],new_fsm_no_copy[-1])
    ll_no_copy = p_corpus(parsed_test_no_copy) # log likelihood (test | grammar)
    print ("No Copy LL test corpus: %.2f"%ll_no_copy)

    # print the difference    
    diff = (ll_copy-ll_no_copy)
    if diff >0:
        print ("\nCopy > No copy")
    else: 
        print ("\nNo Copy > Copy")
    diff = np.abs(diff)
    print ("Difference: %.2f = e^%.3f ~ %f\n"%(diff,diff,np.exp(diff)))
    
    return ll_copy,ll_no_copy
     
windows(corpus,bigrams,ops,bi_ops,3,3)

def windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,windows,start='S',end='F'):
    """
    divides the corpus into "windows" windows;
    for each window, trains on that window and tests on the remainder

    Arguments
    corpus     : string list
    bigrams    : bigram markhov chain
    fsm_copy   : FSM with copy rules
    fsm_no_copy: FSM without copy rules
    windows    : number of windows to divide into (int)
    start      : start state in FSMs
    end        : final state in FSMs

    Returns
    pair
    (list of (ll copy, ll no copy, difference) that came out in favour of the copy grammar,
    list of (ll copy, ll no copy, difference) that came out in favour of the no copy grammar)
    """
    for_copy = [] # a place to store results in favour of the copy grammar
    for_no_copy=[] # a place to store results in favour of the no-copy grammar
    size=len(corpus)
    window_size = size//windows # divide up the corpus into windows this size
    i=0
    while i<windows:
        print ("\nWindow %i of %i"%(i+1,windows))
        #training corpus is one window
        print (i*window_size)
        print (i*window_size+window_size)
        train=corpus[(i*window_size):(i*window_size+window_size)]
        # testing corpus is the other windows together
        test=corpus[:i*window_size]+corpus[i*window_size+window_size:]

        # train the grammars and compare the log likelihoods of the test corpus
        (ll_copy,ll_no_copy)=compare(train,test,bigrams,fsm_copy,fsm_no_copy,n,start,end)
        # add the result to the right list of results
        if ll_copy>ll_no_copy:
            for_copy.append((ll_copy,ll_no_copy,ll_copy-ll_no_copy))
        else:
            for_no_copy.append((ll_copy,ll_no_copy,ll_no_copy-ll_copy))
        i+=1

    print ("\n %i for copy, %i for no copy"%(len(for_copy),len(for_no_copy)))
    return for_copy,for_no_copy



def many_windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,max_windows,start='S',end='F'):
    """
    Divides the corpus into 2 windows, then 3, up to max_windows, and runs windows on it

    Arguments
    corpus     : string list
    bigrams    : bigram markhov chain
    fsm_copy   : FSM with copy rules
    fsm_no_copy: FSM without copy rules
    max_windows: max number of windows to divide into (int)
    start      : start state in FSMs
    end        : final state in FSMs

    Returns
    list of pairs for each number of windows:
    (list of (ll copy, ll no copy, difference) that came out in favour of the copy grammar,
    list of (ll copy, ll no copy, difference) that came out in favour of the no copy grammar)
 
    """
    i=2 # start with 2 windows
    tots = []
    while i<=max_windows:
        # train and test on i windows
        tots.append(windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,i,start='S',end='F'))
        i+=1
    # print a summary of the results
    for w,(for_copy,for_no_copy) in enumerate(tots):
        print ("\n\n%i windows"%(w+2))
        print ("\nFor copy: %i"%(len(for_copy)))
        for (c,nc,dif) in for_copy:
            print (dif)
        print ("\nFor no copy: %i:"%(len(for_no_copy)))
        for (c,nc,dif) in for_no_copy:
            print (dif)

    return tots
        
