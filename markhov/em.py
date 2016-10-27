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


########### DATA STRUCTURES ############

# our probablities are mostly log-transformed due to their being often very small.
# we don't log-transform the expected counts immediately since they're reasonable numbers
# we define some log arithmetic functions in markov.py 

# operations FSAs:
#  no probs:
# We define the operations FSA as a dict with the structure {lhs:{rhs:[emissions]}}
    
# with probs:
# We define the stochastic operations FSA as a dict with the structure {lhs:{rhs:{emission:p}}

#  transitional probabilities / bigrams:

# No probs:                                                                      
# We define a markov chain of words as a dict {a:[b,c,...]} for (a,b),(a,c)... legal bigrams

# With probs:
#  We define a stochastic Markov chain of words as a dict {a:{b:p}} for (a,b),(a,c)... legal bigrams


# corpus : a list of strings

# parsed corpus : a list of (sentence,parse) where parse is a dict encoding bigrams used, 
#route through the operations fsa, and
# transition and state counts for both FSAs
# Note that one sentence can have multiple parses, in which case it appears in multiple parse pairs
# this corpus needs to be kept in order because we define a parallel list of the parses' relative probabilities

#...














############## UTILITY FUNCTIONS #############

def check_fsa(fsa):
    """
    Check if the FSA is valid: if its probabilities for one LHS sum to 1
    """
    ok=True
    for lhs in fsa:
        if len(fsa[lhs])>0:
            tot=log0
            for tr in fsa[lhs]:
                tot=log_add(tot,fsa[lhs][tr])
            if not np.isclose(tot,0.):
                print (lhs)
                ok=False
    return ok



############# PRINTERS ###############

def parsed_corpus2string(corp):
    c=""
    for i,(s,parse) in enumerate(corp):
        c+='\nParse %i:\n'%i
        c+='Sentence: %s\n'%s
        c+='bigrams: %s\n'%' '.join(parse['bis'])
        c+='Q: %s\n'%' '.join(parse['rt'][0])
        c+='E: %s\n'%' '.join(parse['rt'][1])
        c+='SC:\n'
        for x in parse['sc']:
            c+='  %s %i\n'%(x,parse['sc'][x])
        c+='TC:\n'
        for (lhs,(rhs,e)) in parse['tc']:
            c+='  %s %s %s %i\n'%(lhs,e,rhs,parse['tc'][(lhs,(rhs,e))])
        c+='UC:\n'
        for x in parse['uc']:
            c+='  %s %i\n'%(x,parse['uc'][x])
        c+='BC:\n'
        for x in parse['bc']:
            c+='  %s %i\n'%(' '.join(x),parse['bc'][x])

    return c



################# EXPECTED COUNTS ######################

############# counts ################

def scs(qs):
    """ Gets the number of occurrences of each state in qs/each unigram in bis """
    ss={}
    for q in qs:
        ss[q]=ss.get(q,0)+1
    return ss
    

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
        ts[(a,b)]=ts.get((a,b),0)+1
    return ts




def tcs(route):
    """
    gets the number of occurances of all transitions in route (Q,E)

    Arguments
    route : pair (states, emissions) of a route through the operations FSA

    Returns
    dict of (transition): count
    """
    ## TODO Meaghan: maybe not necessary; this function goes through the list of states multiple times
    ## whereas it can do it in one go.
    (qs,es)=route
    ts={}
    for i in range(1,len(qs)):
        q1,e,q2=qs[i-1],es[i-1],qs[i]
        ts[(q1,(q2,e))]=ts.get((q1,(q2,e)),0)+1
    return ts


########### parse corpus ############

def parse_corpus(corpus,bigrams,fsa,start='S'):
    """
    parse the sentences and create a corpus (s,parses,prob s)

    Arguments
    corpus  : string list
    bigrams : bigram morkhov chain (dict)
    fsa     : operations FSA (dict)

    Returns
    a parsed corpus of pairs (i,parse)
       where i is the index of the sentence in the corpus
       and parse is a dict:
    {s (the string itself),
    bis (bigrams used),
    rt (route),
    tc (trans counts), 
    sc (state counts),
    bc (bigram counts), 
    uc (unigram counts)}

    """
    sents = []
    for i,s in enumerate(corpus):
        parses = markhov.parse(s,bigrams,fsa,start) # parse the sentence
        parses = markhov.clean_parses(parses) # just keep (bis,route)
        for (big,(qs,es)) in parses:
            t_counts=tcs((qs,es)) # calculate transition counts
            s_counts=scs(qs) # calculate state counts
            bigram_counts = bi_counts(big)
            unigrams = scs(big[:-1])
            sents.append((i,
                          {'s':s,'bis':big,'rt':(qs,es),
                           'tc':t_counts, 'sc':s_counts,
                           'bc':bigram_counts, 'uc':unigrams}))
    return sents


def get_p_parses(corpus,fsa,bigrams):
    """
    Given a parsed corpus of "bare parses", i.e. parses that only contain the route taken
    and the state and transition counts, and given a current probability assignment 
    we compute the relative probability of each parse (i.e. the probability of the parse
    given the sentence).

    Specifically, we return a list of probabilities (not-logged) that contains, in order,
    the relative probabilities of the particular parse.

    Arguments
    corpus : pairs of (sentence,parse) where sentence occurs multiple times (there a better way to say this)
    fsa : a rule assignment to the operations FSA
    bigrams : a rule assignment to the bigrams FSA

    Returns
    list of (not log-transformed) relative probabilites of parses
    """
    
    # Given a parsed corpus, compute the probabilities of all the parses
    parse_ps = []
    
    # Keeps a running count of the probability of each sentence
    p_s = {}

    # Compute the probabilities of each parse
    for (s,parse) in corpus:
        # Compute the probability of this parse
        p = markhov.p_parse((parse['bis'],parse['rt']),bigrams,fsa)

        # Update the running total of the sentence probabilities
        p_s[s] = log_add(p_s.get(s,log0),p)

        # Add the parse probability of this sentence
        parse_ps.append(p)

    # we don't want this log-transformed because we're going to combine it with expected counts, which are not log-transformed.
    # we think (hope) the relative probability is large enough that this'll be okay
    return [ np.exp(p-p_s[s]) for ((s,_),p) in zip(corpus,parse_ps) ]



def ll_corpus(parsed_corpus,trans_probs,fsa,start='S',end='F'):
    """
    Calculate the log likelihood of the corpus given the grammar

    Arguments
    parsed_corpus : output of parse_corpus. List of (i,parse) pairs where
                    i is the index of the sentence in the corpus and
                    parse is a dict including
                    bis (bigrams used): a list of strings
                    rt (route taken): a pair of lists of strings (Q,E)
    trans_probs   : the transitional probabilties (dict)
    fsa           : the operations FSA (dict)
    start         : the start category for the FSA
    end           : the end category for the FSA

    Returns
    log likelihood of the corpus given the grammar (float)
                    
    """
    # get the part of the parses we actually need
    just_parses = [(s,(parse['bis'],parse['rt'])) for (s,parse) in parsed_corpus]
    # we make a dict of sentences and their lls.
    # We need to add (log-sum) their parse probabilties to get the sentence probabilities
    # and then multiply (add) all their probabilities together
    # to get the total probability of the corpus.
    lls={}
    # this here is why the parsed_corpus is by index, not corpus string:
    # we don't want to add together the probabilities of all the instances of the same sentence.
    for (s,parse) in just_parses:
        lls[s]=lls.get(s,log0)
        lls[s]= log_add(lls[s],markhov.p_parse(parse,trans_probs,fsa))
    # multiply (add in log-space) the likelihoods of all the sentences
    ll=0.
    for s in lls:
        ll+=lls[s]

    return ll



def expected_state_counts(corpus,parse_probs,fsa,count):
    """
    Compute the expected counts of whatever we ask for

    Arguments
    corpus
    

    count : the thing to count: uc, sc

    Returns
    two dicts of expected counts in the structure of the FSA
    TC: Each rule is paired with a dict where the keys are the sentences and the values are the expected counts of that rule for that sentence
    SC: Each state is paired with a dict where the keys are the sentences and the values are the expected counts of that state for that sentence
    """

    expect = {}
    for lhs in fsa: 
        expect[lhs]={}
        for (s,parse),p_rel in zip(corpus,parse_probs): # p_rel is the relative probability, i.e. the parse probability given the sentence and not log-transformed
            expect[lhs][s]=expect[lhs].get(s,0)+ p_rel*parse[count].get(lhs,0)


    return expect


def expected_transition_counts(corpus,parse_probs,fsa,count):
    """
    Compute the expected counts of whatever we ask for

    Arguments
    corpus
    

    count : the thing to count: bc, tc

    Returns
    two dicts of expected counts in the structure of the FSA
    TC: Each rule is paired with a dict where the keys are the sentences and the values are the expected counts of that rule for that sentence
    SC: Each state is paired with a dict where the keys are the sentences and the values are the expected counts of that state for that sentence
    """

    expect = {}
    for lhs in fsa: 
        expect[lhs]={}
        for rhs in fsa[lhs]:
            expect[lhs][rhs]={}
            for (s,parse),p_rel in zip(corpus,parse_probs): # p_rel is the relative probability, i.e. the parse probability given the sentence and not log-transformed
                expect[lhs][rhs][s]=expect[lhs][rhs].get(s,0)+ p_rel*parse[count].get((lhs,rhs),0)


    return expect



############## MAXIMISATION: UPDATE THE GRAMMAR #############

def update(expect_sc,expect_tc,fsa):
    """
    given the expected counts, update the FSA
    """

    # using the structure of the original FSA, 
    # we make new rule probabilities based on the expected counts we calculated
    new_fsa = {}
    for lhs in fsa: # go through the FSA
        new_fsa[lhs]={}
        for tr in fsa[lhs]:
            new_fsa[lhs][tr]={}
                # we make a new prob for this rule based on the ratio of expected counts to the total expected counts for this LHS
                # this will only work if we visited this LHS at all.
            if sum(expect_sc[lhs].values())>0: 
                new_fsa[lhs][tr]= log(sum(expect_tc[lhs][tr].values())/sum(expect_sc[lhs].values()))
            else: # otherwise we divide the probability up evenly. (We could also do it randomly.)
                n_rhs = len(fsa[lhs])
                new_fsa[lhs][tr]=log(1./n_rhs) # 1/n_rhs for this state
                    
    return new_fsa




############# EM #################

def smooth(fsa,sc):
    """
    Smooths FSA probabilities so that no rules have probability 0.

    Specifically, we reduce the probabilities of the rules by the given (sc) amount of probability
    and then distribute it over all rules.

    Arguments
    fsa : a probabilistic finite state machine (dict)
    sc  : smoothing contant (float), not log-transformed

    """
    new_fsa={}
    one_minus_sc=log(1-sc)
    for lhs in fsa:
        new_fsa[lhs]={}
        # each rule set has a probability sum of 1.
        # we divide the existing probabilities by the smoothing contant
        # then we add an equal portion of it to all the rules
        if len(fsa[lhs])>0: # only do this for non-final states
            sc_part = log(sc/len(fsa[lhs])) # we add this to all the rules
            for rhs in fsa[lhs]:
                p=fsa[lhs][rhs]
                if p>log0: # the ones that actually have probability have their probability multiplied by (1-sc)
                    new_fsa[lhs][rhs]=log_add(p+one_minus_sc,sc_part) # we're in log space so we add
                else:  # the ones with no probability just get their new probability added
                    new_fsa[lhs][rhs]=log_add(p,sc_part) 
    assert check_fsa(new_fsa) , "smoothing produced a bad PFSA"
    return new_fsa

    



def initialise(fsa):
    """
    initialise probs to random values

    Arguments
    trans : the bigram transitions (dict)

    Returns
    the bigram transitions with random probabilties added, summing to 1 for each LHS
    log-transformed

    """
 
    # we make a new FSA with probabilities
    fsa_probs = {}
    for a in fsa: #go through the transitions
        fsa_probs[a]={}
        #get the number of transitions from this state
        n= len(fsa[a])
        #generate a list of random numbers, one for each rule
        probs=[random.random() for i in range(n)]
        # we need these to range from [0,1] so we sum them and divide all by the sum
        tot=sum(probs)
        probs = [p/tot for p in probs]
        #Now we add them to the rules
        for i,b in enumerate(fsa[a]):
            fsa_probs[a][b]=np.log(probs[i])
                                  
    return fsa_probs


    
# set the number of iterations
N_ITERATIONS = 1

def em(corpus,trans,fsa_struct,n=N_ITERATIONS,sc=0.,elaborate=True,start='S',end='F'):
    
    """
    sc is initialised to 0 here so we can smooth only at the end if we want

    """

    # Let's do one iteration of EM

    # pre-parsed corpus (but no probabilities assigned)
    corpus=parse_corpus(corpus,trans,fsa_struct,start='S') 
        
    # starting FSA and bigram transitional probabilities
    fsa = initialise(fsa_struct)
    trans_probs = initialise(trans)

    # store the results as we go
    history = []
    # the pre-training stuff
    iter_0 = {'fsa':fsa,'trans_probs':trans_probs}
    if elaborate:
        iter_0['train_ll']=ll_corpus(corpus,trans_probs,fsa,start,end)

    history.append(iter_0)

    for iteration in range(n):

        # EXPECTATION
        # compute the parse probabilities
        parse_ps = get_p_parses(corpus,fsa,trans_probs)

        # compute the expected state and transition counts given our current prob distribution
        scs = expected_state_counts(corpus,parse_ps,fsa_struct,'sc')
        tcs = expected_transition_counts(corpus,parse_ps,fsa_struct,'tc')
        ucs = expected_state_counts(corpus,parse_ps,trans,'uc')
        bcs = expected_transition_counts(corpus,parse_ps,trans,'bc')

        
        # MAXIMISATION
        # compute the updated rule probabilities
        fsa         = smooth(update(scs,tcs,fsa_struct),sc)
        trans_probs = smooth(update(ucs,bcs,trans),sc)

        assert check_fsa(fsa), "New FSA isn't valid"
        assert check_fsa(trans_probs), "New Trans Probs isn't valid"

        result = {"fsa":fsa,
                  "scs":scs,
                  "tcs":tcs,
                  'ucs':ucs,
                  'bcs':bcs,
                  "trans_probs":trans_probs,
                  "parse_ps":parse_ps}
        
        if elaborate:
            # add the LL of the corpus we just trained on
            result['train_ll']=ll_corpus(corpus,trans_probs,fsa,start,end)           

        history.append(result)


    return history,corpus



SC=0.01

def em_train(train,test,trans,fsm,n=N_ITERATIONS,presmooth=True,sc=SC,elaborate=True,start='S',end='F'):
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
    if presmooth:
        em_sc=sc
    else:
        em_sc=0.

    # train
    history,train=em(train,trans,fsm,n,em_sc,elaborate,start)
    # extract values from last training cycle
    new_parse_ps = history[-1]['parse_ps']
    new_fsa = history[-1]['fsa']
    new_trans = history[-1]['trans_probs']
    
    #smooth if nec
    if not presmooth:
        new_trans=smooth(new_trans,sc)
        new_fsa=smooth(new_fsa,sc)

    # calculate ll train with smoothing
    ll_train = ll_corpus(train,new_trans,new_fsa,start,end)
    print ("LL training corpus: %.2f"%ll_train)
    
    # parse the test corpus with the trained grammar
    parsed_test = parse_corpus(test,new_trans,new_fsa)
    # calculate ll test with smoothing    
    ll_test = ll_corpus(parsed_test,new_trans,new_fsa,start,end)
    print ("LL testing corpus: %.2f"%ll_test)

    # add in all lls (smoothed)
    for i in range(len(history)):
        ll=ll_corpus(parsed_test,history[i]['trans_probs'],history[i]['fsa'],start,end)
        history[i]['test_ll']=ll

    # add the smoothed version to the history
    if not presmooth:
        smoothed={}
        for key in history[-1]:
            smoothed[key]=history[-1][key]
        smoothed['fsa']=new_fsa
        smoothed['trans_probs']=new_trans
        smoothed['test_ll']=ll_test
        smoothed['train_ll']=ll_train
        history.append(smoothed)    

    return ll_test,train,parsed_test,history
        

def compare(train,test,bigrams,fsm_copy,fsm_no_copy,n=N_ITERATIONS,presmooth=True,sc=SC,start='S',end='F',verbose=False):
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
    ll_copy,parsed_train,parsed_test,history_copy = em_train(train,test,bigrams,fsm_copy,n,presmooth,sc,start,end)
    #print ("Copy LL test corpus: %.2f"%ll_copy)

    print ("\nNo Copy")
    # train no copy grammar
    # we only need to EM once
    ll_no_copy,parsed_train,parsed_test,history_no_copy = em_train(train,test,bigrams,fsm_no_copy,2,presmooth,sc,start,end)
    #print ("No Copy LL test corpus: %.2f"%ll_no_copy)

    # print the difference    
    diff = (ll_copy-ll_no_copy)
    if diff >0:
        print ("\nCopy > No copy")
    else: 
        print ("\nNo Copy > Copy")
    diff = np.abs(diff)
    print ("Difference: %.2f = e^%.3f ~ %f\n"%(diff,diff,np.exp(diff)))
    
    return (ll_copy,history_copy),(ll_no_copy,history_no_copy)
     

def windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,w,presmooth=True,sc=SC,start='S',end='F'):
    """
    divides the corpus into "windows" windows;
    for each window, trains on that window and tests on the remainder

    Arguments
    corpus     : string list
    bigrams    : bigram markhov chain
    fsm_copy   : FSM with copy rules
    fsm_no_copy: FSM without copy rules
    n          : number of EM iterations
    windows    : number of windows to divide into (int)
    presmooth  : whether to smooth at each step or only at the end
    sc         : smoothing constant: the amount of probability to distribute over rhss
    start      : start state in FSMs
    end        : final state in FSMs

    Returns
    pair
    list of ((ll copy, copy history),(ll no copy, no copy history)), one for each window 
    """
    results = [] # a place to store results

    size=len(corpus)
    window_size = size//w # divide up the corpus into windows this size
    i=0
    while i<w:
        print ("\nWindow %i of %i"%(i+1,w))
        #training corpus is one window
        print ((i*window_size),(i*window_size+window_size))
        train=corpus[(i*window_size):(i*window_size+window_size)]
        # testing corpus is the other windows together
        test=corpus[:i*window_size]+corpus[i*window_size+window_size:]

        # train the grammars and compare the log likelihoods of the test corpus
        # add the result to the right list of results
        results.append(compare(train,test,bigrams,fsm_copy,fsm_no_copy,n,presmooth,sc,start,end))
        i+=1

    return results


def iter_windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,w,runs,presmooth=True,sc=SC,start='S',end='F'):
    """
    Runs windows multiple times, re-initialising the grammar probabilities each time

    Arguments
    corpus     : string list
    bigrams    : bigram markhov chain
    fsm_copy   : FSM with copy rules
    fsm_no_copy: FSM without copy rules
    n          : number of EM iterations
    windows    : number of windows to divide into (int)
    runs       : number of times to do the whole thing 
    presmooth  : whether to smooth at each step or only at the end
    sc         : smoothing constant: the amount of probability to distribute over rhss
    start      : start state in FSMs
    end        : final state in FSMs

    Returns
    pair
    list of lists of ((ll copy, copy history),(ll no copy, no copy history)), one for each run

    """

    results=[]
    i=0
    while i<=runs:
        print ("\nRun %i of %i"%(i,runs))
        results.append(windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,w,presmooth,sc,start,end))
        i+=1

    return results



def many_windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,max_windows,presmooth=True,sc=SC,start='S',end='F'):
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
        print ("\nDividing corpus into %i windows"%i)
        # train and test on i windows
        tots.append(windows(corpus,bigrams,fsm_copy,fsm_no_copy,n,i,presmooth,sc,start='S',end='F'))
        i+=1
    # print a summary of the results

    return tots
        
