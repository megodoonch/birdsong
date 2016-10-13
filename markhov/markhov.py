"""
Two Marknov chains, one with bigrams and the other with merge, copy, and clear

Build two parallel strings, string and buffer. Copy and clear buffer.

Meaghan Fowlie and Floris van Vugt
last edited: Oct 11,2016
"""

import numpy as np
import random

log0=float('-inf')


def log_add(logx,logy):
    """This adds two log-transformed variables,
    taking care of the underflow that you usually find when you do this

    Arguments: 
    logx : float
    logy : float"""

    log0 = float('-inf')
    if logx==log0 and logy==log0: # avoid nan
        return log0
    # First, make X the maximum
    if (logy > logx):
        logx,logy = logy,logx
        #temp = logx
        #logx = logy
        #logy = temp

    # How far "down" is logY from logX?
    negdiff = logy - logx
    if negdiff < -30: # If it's small, we can just ignore logY altogether (it won't make much of a difference)
        return logx
        # However, in my case I can maybe keep it in because it will just become zero in the sum below.

    # Otherwise, use some simple algebra to stay in the log domain
    # (i.e. here we use log(X)+log(Y) = log(X)+log(1.0+exp(log(Y)-log(X)))
    return logx + np.log(1.0 + np.exp(negdiff))

#### OPERATIONS PFSAs #######

def ops_log(ops):
    for a in ops:
        for b in ops[a]:
            for w in ops[a][b]:
                ops[a][b][w]=np.log(ops[a][b][w])
    return ops


ops = {'S':{'COPY':{'mg':1.}}, # from start we have to merge
       'COPY':{'COPY':{'mg':0.4,'copy':0.1}, # this state is the state in which the last "special" operation was *not* Clear. Either we've done none or the last was copy. From here we can do everything including end
               'CLEAR_S':{'clear':0.1}, # go here to clear the buffer
               'F':{'end':0.5} # go here to end
           },
       'CLEAR_S':{'CLEAR':{'mg':1.}}, # this is where we've just cleared. Buffer is empty so you can only Merge
       'CLEAR':{'CLEAR':{'mg':0.5}, # the last special op was Clear so we can Copy or Merge.
                'COPY':{'copy':0.5} # if we Copy, the last special op was Copy so go to COPY
            },
       'F':{} #final state
   }

ops=ops_log(ops)

bi_ops = {'S':{'S':{'mg':0.5},
               'F':{'end':0.5}
           },
          'F':{}
      }

bi_ops=ops_log(bi_ops)



##### PRINT #####

def parse2string(parse):
    (bis,buf,ops,cleared,inds)=parse
    return "\nbigrams: %s\nbuffer: %s\noperations: %s\ncleared: %s\ni: %s"%(' '.join(bis),' '.join(buf),' '.join(ops),cleared,' '.join([str(i) for i in inds]))



##### choosing the move #######

def next_step(current,chain):
    """
    Chooses next step based on probabilities in chain
    Returns step and its prob
    Arguments:
    current : current state in markhov chain
    chain   : Markhov chain
    """
    p=np.log(random.random()) # random prob for deciding next move
    #print(p)
    nexts = chain[current]
    current_p = log0 # set the floor to 0
    for o in nexts.keys():
        prob=nexts[o] # get next prob
        #print("%s: (%f,%f)"%(o, current_p,prob+current_p))
        if p>=current_p and p < log_add(prob,current_p): # if the random prob falls between the last prob and this one, we've found our result
            return o
        else:
            current_p=log_add(prob,current_p) # if not, set the floor for the next interval to check
    return ("no transition")
            


########### GENERATE A SENTENCE ##############

def step(state,ops):
    """
    Chooses a next state in ops PFSM based on probabilities in machine
    """
    p=np.log(random.random()) # random prob for deciding next move
    # we'll look for an interval in the probs of out-arrows in which p falls. 
    current_p=log0 # this is the bottom of the first interval
    nexts=ops[state] # possible next states

    for st in nexts:
        #print(st)
        for op in nexts[st]:
            #print (op)
            p_next=nexts[st][op] # the prob of this arrow
            #print (p_next)
            #print ("is %0.5f < %0.5f < %0.5f?"%(current_p,p,p_next))
            if p>=current_p and p < log_add(p_next,current_p): # if the random prob falls between the last prob and this one, we've found our result
                #print (st,op,p)
                return (st,op,p_next) # this is it!
            else:
                current_p=log_add(p_next,current_p) #otherwise, move bottom of interval up



def generate_ops(ops):
    """
    Generates a string of operation based on ops PFSM
    """
    state='S'
    out=[]
    p=0.
    next_step=step(state,ops)
    while next_step!=None:
        out.append(next_step[1])
        p=p+next_step[2]
        next_step=step(next_step[0],ops)
    return out,p


def generate_string(op_string,bigrams):
    """Generates string based on operation string and bigrams markhov chain"""
    s=["["] # we need a start string to calculate transitional probs for starting
    b=[]
    for op in op_string:
        if op=='mg':
            next_word=next_step(s[-1],bigrams)
            s.append(next_word)
            b.append(next_word)
        elif op=='copy':
            s+=b
            b+=b
        elif op=='clear':
            b=[]
        elif op=='end':
            s.append(']')
            print (' '.join(s))
            return s
        else: 
            print("bad operation name")


##wrapper
def gen(bigrams,ops):
    return generate_string(generate_ops(ops)[0],bigrams)

def gen_corpus(bigrams,ops,n):
    i=0
    corpus=[]
    while i<n:
        corpus.append(gen(bigrams,ops))
    return corpus


######### PARSE ###########

def parse(s,bigrams,verbose=False):
    """
    Parses a string
    Uses an agenda and a list of complete parses.
    We initialise the agenda with a start state,   [['['],[],['mg'],False]
    An agenda item is a list of 4 elements:
    the string so far, the buffer, the operations, and whether the last "special" operation (copy or clear) is clear.
    The last element is a little hack to reduce the amount of pointless buffer-clearing.
    We add onto an agenda item until it is a complete parse, in which case we move it to the output, parses
    Whenever there is a choice of two valid moves, we copy the agenda item and try both
        
    Arguments:
    s       : the string to be parsed -- string
    bigrams : bigram markhov chain
    """

    #s = s.split(' ') # make the string into a list
    #s=['[']+s+[']']
    n=len(s)#-1
    #possible = n>0 and s[0]=='['
    #if possible: # only sentence with at least [ and ] are going to be grammatical
    agenda = [ (['['],[],[],False,[0,1]) ]  # initialise the agenda
    #else:
    #    return (False,[])
    parses = [] # this will be our output
    while len(agenda)>0: # keep parsing as long as we have incomplete parses
        # try to extend all partial parses in all possible ways
        if verbose: print ("\nAgenda to copy and clear: %i"%len(agenda))
        for parse in agenda:            
            if verbose: print("\nparse to copy/clear: %s"%parse2string(parse))
            (bis,buf,ops,cleared,indices)=parse
            if verbose: print (parse)
            i=indices[-1]
            if i==n: # move a complete parse over to the output
                agenda.remove(parse)
                new_bis=bis[:]
                #new_bis.append(']')
                new_buf=buf[:]
                new_ops=ops[:]
                new_ops.append('end')
                new_indices=indices[:]
                final_parse = (new_bis,new_buf,new_ops,cleared,new_indices)
                parses.append(final_parse)

            else:
                if len(buf)>0: # if buffer not empty
                    ## try clearing the buffer
                    if verbose: print ("\n Clear")
                    if not cleared:  #i if last special op not clear
                        # this is to make copies of the lists inside the list
                        new_bis=bis[:]
                        new_ops=ops[:]
                        new_ops.append('clear') # copy the list of operations
                        new_indices=indices[:]
                        new_parse=(new_bis,[],new_ops,True,new_indices)
                        if verbose: print ("new parse: %s"%parse2string(new_parse))
                        agenda.append(new_parse) # add this new parse to the agenda

                    #Try to Copy
                    if verbose: print ("\n Copy")
                    if verbose: print ("buffer ",buf)
                    copy_end = i+len(buf)
                    if verbose: print ("copy? ", s[i : copy_end])
                    # if the buffer is the same as the next part of the sentence
                    if i+len(buf) <= n and parse[1]==s[i : copy_end]: 
                        new_bis=bis[:]
                        new_ops=ops[:]
                        new_ops.append('copy') # copy the list of operations and add copy
                        new_buf=buf[:]+buf[:] #add copy to buffer
                        new_is=indices[:]
                        new_is.append(copy_end)
                        new_parse=(new_bis,new_buf,new_ops,False,new_is)
                        if verbose: print ("new parse : %s"%parse2string(new_parse))
                        agenda.append(new_parse) # add the new parse to the agenda

        if verbose: print ("\nAgenda to merge: %i"%len(agenda))
        if verbose: 
            for parse in agenda:
                print(parse2string(parse))
        for parse in agenda:            
            if verbose: print("\nparse to merge: %s"%parse2string(parse))
            (bis,buf,ops,cleared,indices)=parse
            if verbose: print (parse)
            i=indices[-1]
            if i==n: # move a complete parse over to the output
                agenda.remove(parse)
                new_bis=bis[:]
                #new_bis.append(']')
                new_buf=buf[:]
                new_ops=ops[:]
                new_ops.append('end')
                new_indices=indices[:]
                final_parse = (new_bis,new_buf,new_ops,cleared,new_indices)
                parses.append(final_parse)
            else:
       
                # Try to Merge
                if verbose: 
                    print ("\n Merge")
                    print (s[i])
                    print(s[i-1])
                if s[i] in bigrams[s[i-1]]:
                    bis.append(s[i]) #string
                    buf.append(s[i]) #buffer
                    ops.append('mg') #list of ops
                    indices.append(i+1) # advance by 1
                    if verbose: print ("merged parse: %s"%parse2string(parse))
                else:
                    return (False,parses) # if this isn't a legal transition, this sentence is not grammatical

    return (True,parses)




def probability_bigrams(s,bigrams):
    """
    Calculates the probability of the sentence based just on the bigrams
    """
    s=s.split(' ')
    p=0.
    for i in range(1,len(s)):
        p+=bigrams[s[i-1]][s[i]]
    return p


def check_bigrams(s,bigrams):
    """
    Calculates the probability of the string based just on the bigrams

    Arguments:
    s       : sequence of merged words. If there's no copying this is the whole string; otherwise some is missing
    bigrams : markhov chain of transitions implemented as dict
    """
    p=0.
    for i in range(1,len(s)):
        if s[i-1] in bigrams and s[i] in bigrams[s[i-1]]:
            p+=bigrams[s[i-1]][s[i]]
        else: return (False,log0)
    return (True,p)





def check_ops(op_list,ops,verbose=False):
    """
    Checks the validity of a sequence of operations. 
    Returns probability of that sequence

    Arguments:
    op_list : sequence of operations chosen from mg,copy,clear,end
    ops     : PFSM of operations implemented as dict
    """
    p=0.
    state='S'
    valid=True # valid parse
    while len(op_list)>0 and valid: # we stop if we run out of operations or we realise this isn't even a valid parse
        if verbose: print("Current state: %s"%state)
        if verbose: print (op_list)
        op=op_list[0] # take the first operation
        nexts=ops[state] # find the out-arrows
        next_state=None # initialise next state
        new_p=log0 # initialise prob of transition
        for st in nexts: # look through out arrows
            if verbose: print (st)
            if op in nexts[st].keys(): # if this is the right arrow
                new_p=nexts[st][op] # prob of transition
                next_state=st # new state
                break # we're done here
        p+=new_p # multiply in new prob
        valid=next_state!=None # if we didn't find a state to transition to, the parse is invalid
        op_list=op_list[1:] # onto the next operation. I think I'm being overly OCamly here.
        state=next_state # onto the next state

    return (valid and state=='F',p) # it's valid if we didn't run into a problem and we end in a final state


def prob_parse(parse,bigrams,ops):
    """
    Finds the probability of a single parse given both machines

    Arguments:
    parse   : (bigram string, buffer, operations used, last special op was Clear, list of indices we considered)
    bigrams : Markhov chain of alphabet (dict)
    ops     : PFSM of operations (dict)
    """
    bis = parse[0]
    op_list = parse[2]
    #print(check_bigrams(bis,bigrams))
    #print(check_ops(op_list,ops))
    return check_bigrams(bis,bigrams)[1] + check_ops(op_list,ops)[1] # multiply probs


def prob_string(s,bigrams,ops,verbose=False):
    """
    finds the total probability of a string given the operations and bigrams

    s      : sentence (string list)
    bigrams: markhov chain of alphabet members (dict)
    ops    : PFSM of operations (dict)
    """
    parses=parse(s,bigrams)
    p=log0
    if not parses[0]:
        print ("not a valid sentence")
        return log0
    for i,par in enumerate(parses[1]):
        new_p=prob_parse(par,bigrams,ops)
        if verbose: print("parse %i: %.5f"%(i,new_p))
        p = log_add(p,new_p)
    print ("total prob: %.5f"%p)
    return p
