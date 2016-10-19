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


# def add_counts(tot,p,tcounts,scounts,tr):#,p_s):
#     """
#     add counts times prob parse

#     Arguments
#     tot     : the running total we're adding to
#     p       : prob of parse
#     tcounts : transition counts
#     scounts : state counts
#     tr     : the transition. Note the first element is the state, whether it's (q1,e,q2) or (a,b)
    
#     Returns
#     the new running total (float)
#     """
#     print ("%.2f + %.2f x %.2f / %.2f=%.2f"%(np.exp(tot), np.exp(p),tcounts[tr],scounts[tr[0]],np.exp(log_add(tot, p+log(tcounts[tr])-log(scounts[tr[0]])))))

#     return log_add(tot, p+log(tcounts[tr])-log(scounts[tr[0]]))
     

  
# def update_automata(corpus,bigrams,fsa):
#     """
#     updates the bigram chain and operations fsa
    
#     Arguments
#     corpus : a parsed corpus of triples (s,parses,prob)
#               where parses is a list of
#               (bis,route,prob,state counts, trans counts,
#               bigram counts, unigram counts)
#     bigrams : bigram morkhov chain (dict)
#     fsa     : operations FSA (dict)

#     returns
#     updated bigrams, updated fsa

#     """
#     new_fsa = copy.deepcopy(fsa) #copy the ops FSA
#     for lhs in new_fsa: # go through the FSA
#         #print (lhs)
#         for rhs in new_fsa[lhs]:
#             #print (rhs)
#             for e in new_fsa[lhs][rhs]:
#                 #print (e)
#                 tot_tc = log0 # transition counts
#                 tot_sc = log0 #state counts
                
#                 for (s,parses,p_s) in corpus: # go through parsed corpus
#                     #print (s)
#                     for (bis,route,p,sc,tc,bc,uc) in parses:
#                         #print (tc)
#                         #print (sc)
#                         if (lhs,e,rhs) in tc: #look for current transition in counts
#                             #print (lhs,e,rhs)
#                             # add in p(parse | sent) * TC 
#                             tot_tc = add_counts(tot_tc,p,p_s,tc,(lhs,e,rhs)) 
#                         if lhs in sc: # look for current state in counts
#                             #print (lhs)
#                             # add in p(parse | sent) * SC
#                             tot_sc = add_counts(tot_sc,p,p_s,sc,lhs)
#             #if tot_tc==log0 or tot_sc==log0:
#             #    print (tot_tc,tot_sc)
#             new_fsa[lhs][rhs][e]=tot_tc-tot_sc # update prob

#     new_bis = copy.deepcopy(bigrams) # copy bigrams
#     for a in bigrams: #go through the chain
#         for b in bigrams[a]:
#             tot_bc=log0 #bigram counts
#             tot_uc=log0 # unigram counts
#             for (s,parses,p_s) in corpus: # go through the corpus
#                 for (bis,route,p,sc,tc,bc,uc) in parses:
#                     if (a,b) in bc: # look for bigram in counts
#                         tot_bc=add_counts(tot_bc,p,p_s,bc,(a,b))
#                     if a in uc: # look for unigram in counts
#                         tot_uc = add_counts(tot_uc,p,p_s,uc,a)
#             new_bis[a][b]=tot_bc-tot_uc # new prob
            
#     return new_bis,new_fsa





# def update_automata(corpus,bigrams,fsa):
#     """
#     updates the bigram chain and operations fsa
    
#     Arguments
#     corpus : a parsed corpus of triples (s,parses,prob)
#               where parses is a list of
#               (bis,route,prob,state counts, trans counts,
#               bigram counts, unigram counts)
#     bigrams : bigram morkhov chain (dict)
#     fsa     : operations FSA (dict)

#     returns
#     updated bigrams, updated fsa

#     """
#     new_fsa = copy.deepcopy(fsa) #copy the ops FSA
#     for lhs in new_fsa: # go through the FSA
#         #print (lhs)
#         for rhs in new_fsa[lhs]:
#             #print (rhs)
#             for e in new_fsa[lhs][rhs]:
#                 #print (e)
#                 new_prob=log0
#                 print (lhs,e,rhs)

#                 for (s,parses,p_s) in corpus: # go through parsed corpus
#                     #print (s)
#                     # add up all the TCs/SCs for the parses of this sentence
#                     #new_p_this_s=log0
#                     for (bis,route,p,sc,tc,bc,uc) in parses:
#                         #print (tc)
#                         #print (sc)
#                         if (lhs,e,rhs) in tc: # if this parse has this rule
#                             # add in this count ratio, times p(parse)
#                             new_prob = add_counts(new_prob,p,tc,sc,(lhs,e,rhs))#,p_s)
#                             #new_p_this_s = add_counts(new_p_this_s,p,tc,sc,(lhs,e,rhs))
#                     # when you've got all the parses for this sentence processed, 
#                     #divide by the prob of the sentence 
#                     #and add the result into the new prob for the rule
#                     #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
#                     #print (lhs,e,rhs,new_prob)
#             # when you're through the corpus, update prob for this rule
#             if np.isnan(new_prob): print ("uh oh!", lhs,e,rhs)
#             new_fsa[lhs][rhs][e]=new_prob 

#     new_bis = copy.deepcopy(bigrams) # copy bigrams
#     for a in bigrams: #go through the chain
#         for b in bigrams[a]:
#             new_prob=log0
#             print (a,b)

#             for (s,parses,p_s) in corpus: # go through the corpus
#                 #new_p_this_s=log0
#                 for (bis,route,p,sc,tc,bc,uc) in parses:
#                     if (a,b) in bc: # look for bigram in counts
#                         # add in this count ratio, times p(parse)
#                         new_prob = add_counts(new_prob,p,bc,uc,(a,b))#,p_s)
#                         #new_p_this_s = add_counts(new_p_this_s,p,bc,uc,(a,b))
#                 #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
#             new_bis[a][b]=new_prob # new prob
            
#     return new_bis,new_fsa



def p_corpus(parsed_corpus):
    """
    Computes the log likelihoods of the corpus given the latest grammar
    These are already computed and stored in the parsed corpus

    """
    return sum([p for (s,parses,p) in parsed_corpus])


def sum_p_corpus(parsed_corpus):
    return  log_sum([p for (s,parses,p) in parsed_corpus])



# def update_automata(corpus,bigrams,fsa):
#     """
#     updates the bigram chain and operations fsa
    
#     Arguments
#     corpus : a parsed corpus of triples (s,parses,prob)
#               where parses is a list of
#               (bis,route,prob,state counts, trans counts,
#               bigram counts, unigram counts)
#     bigrams : bigram morkhov chain (dict)
#     fsa     : operations FSA (dict)

#     returns
#     updated bigrams, updated fsa

#     """
#     p_c=sum_p_corpus(corpus)
#     #print ("prob corp = %f"%np.exp(p_c))
#     new_fsa = copy.deepcopy(fsa) #copy the ops FSA
#     for lhs in new_fsa: # go through the FSA
#         print (lhs)
#         for rhs in new_fsa[lhs]:
#             #print (rhs)
#             for e in new_fsa[lhs][rhs]:
#                 #print (e)
#                 new_prob=log0
#                 #print (lhs,e,rhs)

#                 for (s,parses,p_s) in corpus: # go through parsed corpus
#                     #print (s)
#                     # add up all the TCs/SCs for the parses of this sentence
#                     #new_p_this_s=log0
#                     for (bis,route,p,sc,tc,bc,uc) in parses:
#                         #print (tc)
#                         #print (sc)
#                         if (lhs,e,rhs) in tc: # if this parse has this rule
#                             # add in this count ratio, times p(parse)
#                             new_prob = add_counts(new_prob,p,tc,sc,(lhs,e,rhs))#,p_s)
#                             #new_p_this_s = add_counts(new_p_this_s,p,tc,sc,(lhs,e,rhs))
#                     # when you've got all the parses for this sentence processed, 
#                     #divide by the prob of the sentence 
#                     #and add the result into the new prob for the rule
#                     #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
#                     #print (lhs,e,rhs,new_prob)
#             # when you're through the corpus, update prob for this rule
#             print ("new prob for %s %s %s: %f - %f = %f"%(lhs,e,rhs,new_prob,p_c,new_prob-p_c))
#             if np.isnan(new_prob): print ("uh oh!", lhs,e,rhs)
#             new_fsa[lhs][rhs][e]= (new_prob - p_c)
#             #print (new_fsa[lhs][rhs][e])
#     #print (new_fsa)

#     new_bis = copy.deepcopy(bigrams) # copy bigrams
#     for a in bigrams: #go through the chain
#         for b in bigrams[a]:
#             new_prob=log0
#             #print (a,b)

#             for (s,parses,p_s) in corpus: # go through the corpus
#                 #new_p_this_s=log0
#                 for (bis,route,p,sc,tc,bc,uc) in parses:
#                     if (a,b) in bc: # look for bigram in counts
#                         # add in this count ratio, times p(parse)
#                         new_prob = add_counts(new_prob,p,bc,uc,(a,b))#,p_s)
#                         #new_p_this_s = add_counts(new_p_this_s,p,bc,uc,(a,b))
#                 #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
#             new_bis[a][b]=new_prob - p_c # new prob
            
#     return new_bis,new_fsa


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
                    #new_p_this_s=log0
                    for (bis,route,p,sc,tc,bc,uc) in parses:
                        #print (tc)
                        #print (sc)
                        if (lhs,e,rhs) in tc: # if this parse has this rule
                            # add in counts, times p(parse)
                            tot_tc=log_add(tot_tc, p + log(tc[(lhs,e,rhs)]))
                            if lhs in sc:
                                tot_sc=log_add(tot_sc, p + log(sc[lhs]))
                            #new_p_this_s = add_counts(new_p_this_s,p,tc,sc,(lhs,e,rhs))
                    # when you've got all the parses for this sentence processed, 
                    #divide by the prob of the sentence 
                    #and add the result into the new prob for the rule
                    #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
                    #print (lhs,e,rhs,new_prob)
            # when you're through the corpus, update prob for this rule
            if verbose: print ("new prob for %s %s %s: %f - %f = %f"%(lhs,e,rhs,tot_tc,tot_sc,tot_tc-tot_sc))
            if tot_sc!=log0: # only make a new prob if we used this rule
                new_fsa[lhs][rhs][e]= tot_tc-tot_sc
            else: new_fsa[lhs][rhs][e]=fsa[lhs][rhs][e]
            #print (new_fsa[lhs][rhs][e])
    #print (new_fsa)

    new_bis = copy.deepcopy(bigrams) # copy bigrams
    for a in bigrams: #go through the chain
        for b in bigrams[a]:
            #print (a,b)
            tot_bc=log0
            tot_uc=log0
            for (s,parses,p_s) in corpus: # go through the corpus
                #new_p_this_s=log0
                for (bis,route,p,sc,tc,bc,uc) in parses:
                    if (a,b) in bc: # look for bigram in counts
                        # add in counts, times p(parse)
                        tot_bc=log_add(tot_bc, p + log(bc[(a,b)]))
                        tot_uc=log_add(tot_uc, p + log(uc[a]))
                #new_prob = log_add(new_prob, new_p_this_s)#-p_s)
            if verbose: print ("new prob for %s %s: %f - %f = %f"%(a,b,tot_bc,tot_uc,tot_bc-tot_uc))
            if tot_uc!=log0: # only make a new prob if we used this rule
                new_bis[a][b]=tot_bc-tot_uc # new prob
            else: new_bis[a][b]=bigrams[a][b]
            
    return new_bis,new_fsa



def check_fsa(fsa):
    ok=True
    for lhs in fsa:
        tot=log0
        for rhs in fsa[lhs]:
            for e in fsa[lhs][rhs]:
                tot=log_add(tot,fsa[lhs][rhs][e])
        ok = np.isclose(tot,0.)
    return ok


def em_main(parsed_corpus,bigrams,fsm,n,start='S',end='F'):
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
    print ("initial LL: %.2f"%p_corpus(parsed_corpus)) # LL (corpus | grammar)
    print ("iteration %i"%i)
    if n>0: # if we said 0 we'd better not do anything
        # first we use the given corpus and automata
        new_c,new_bis,new_fsa=inner(parsed_corpus,bigrams,fsm)
        corpora.append(new_c)
        bigram_chains.append(new_bis)
        ops_fsas.append(new_fsa)
        print ("LL: %.2f"%p_corpus(new_c)) # LL (corpus | new grammar)

    i+=1
    while i<n:
        print ("iteration %i"%i)
        # now we have new automata to iterate over
        new_c,new_bis,new_fsa=inner(new_c,new_bis,new_fsa)
        corpora.append(new_c)
        bigram_chains.append(new_bis)
        ops_fsas.append(new_fsa)
        print ("LL: %.2f"%p_corpus(new_c)) # LL (corpus | new grammar)
        i+=1
    
    return corpora,bigram_chains,ops_fsas



def em(corpus,bigrams,fsm,n,start='S',end='F'):
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
    return em_main(parsed_corpus,bigrams,fsm,n,start,end)
