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

# def ops_log(ops):
#     for a in ops:
#         for b in ops[a]:
#             for w in ops[a][b]:
#                 ops[a][b][w]=np.log(ops[a][b][w])
#     return ops



# ops = {'S':{'NotCL':{'mg':1.}}, # from start we have to merge
#        'NotCL':{'NotCL':{'mg':0.25,'copy':0.25}, # this state is the state in which the last "special" operation was *not* Clear. Either we've done none or the last was copy. From here we can do everything including end
#                 'CLEAR_S':{'clear':0.25}, # go here to clear the buffer
#                'F':{'end':0.25} # go here to end
#            },
#        'CLEAR_S':{'CLEAR':{'mg':1.}}, # this is where we've just cleared. Buffer is empty so you can only Merge
#        'CLEAR':{'CLEAR':{'mg':0.5}, # the last special op was Clear so we can Copy or Merge.
#                 'NotCL':{'copy':0.5} # if we Copy, the last special op was Copy so go to COPY
#             },
#        'F':{} #final state
#    }





# ops=ops_log(ops)

# bi_ops = {'S':{'S':{'mg':0.5},
#                'F':{'end':0.5}
#            },
#           'F':{}
#       }

# bi_ops=ops_log(bi_ops)



##### PRINT #####

def parse2string(parse):
    (bis,buf,(qs,ops),k,p)=parse
    return "\nbigrams: %s\nbuffer: %s\nop states: %s\noperations: %s\nk: %i\nprob: %.4f"%(' '.join(bis),' '.join(buf),' '.join(qs),' '.join(ops),k,p)

def parse2tree(parse):
    (bis,buf,(p_qs,p_ops),k,p)=parse
    qs,ops=p_qs[:],p_ops[:]
    qs.reverse() # work from the bottom up
    ops.reverse()
    qs=qs[1:] # we don't want F: this just means end is a terminal
    t=(qs[0],[ops[0]]) # base case
    i=1
    for (q,op) in zip(qs[1:],ops[1:]):
        if op=='mg':
            left = "mg\n%s"%bis[i]
            i+=1
        else:
            left=op
        t=(q,[left,t])
    return t


def parse2dot(parse):
    """
    makes a dot-readable string of nodes and edges

    Arguments
    parse : a parse of the string of the form (bigrams,_,(states visited, operations used),_)

    Returns
    nodes and edges where nodes are pairs of unique names and labels, and edges are pairs of nodes
    """
    (bis,buf,(qs,ops),k,p)=parse
    nodes=[]
    edges=[]
    n=0 # for naming nodes uniquely
    i=1 # for moving through the bigrams
    for q,op in zip(qs[:-2],ops[:-1]):
        nodes.append((n,q)) # mother node
        if op=='mg': # if it's merge, let's get the word too
            left = "mg\n%s"%bis[i]
            i+=1
        else:
            left=op
        nodes.append((n+1,left)) # left daughter is lexical
        edges.append((n,n+1)) # edge to left daughter
        edges.append((n,n+2)) # edge to right daughter
        n+=2 # right daughter is the new mother node

    nodes.append((n,qs[-2])) # the last constituent is unary-branching
    nodes.append((n+1,ops[-1]))
    edges.append((n,n+1))

    return nodes,edges



def dot_output(parse):
    # Make a little dot output for the particular tree
    nodes,edges = parse2dot(parse)

    outp = ""

    outp += "digraph tree {\n node [shape = none; height=0; width=0];\n edge [dir=none];"
    
    for (nodename,nodelabel) in nodes:
        outp += "\t%s [label=\"%s\"]\n"%(nodename,nodelabel)

    for (from_node,to_node) in edges:
        outp += "\t%s -> %s\n"%(from_node,to_node)

    outp+="}\n"
    return outp


def parse2pic(parse,fname,ftype):
    # Makes a dot graph and outputs to pdf
    outp = dot_output(parse)
    f = open('.tmp.dot','w')
    f.write(outp)
    f.close()
    import subprocess
    subprocess.call(['dot','.tmp.dot','-T%s'%ftype,'-o',"%s.%s"%(fname,ftype)])
    # subprocess.call(['rm','.tmp.dot']) # clean up my mess
    
    return




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
    if current in chain:
        nexts = chain[current]
    else:
        return
        
    current_p = log0 # set the floor to 0
    for o in nexts.keys():
        prob=nexts[o] # get next prob
        #print("%s: (%f,%f)"%(o, current_p,prob+current_p))
        if p>=current_p and p < log_add(prob,current_p): # if the random prob falls between the last prob and this one, we've found our result
            return o
        else:
            current_p=log_add(prob,current_p) # if not, set the floor for the next interval to check
    print ("no transition")
    return
            


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
            if next_word==None:
                print ("no transition")
                return
            s.append(next_word)
            b.append(next_word)
        elif op=='copy':
            s+=b
            b+=b
        elif op=='clear':
            b=[]
        elif op=='end':
            print (' '.join(s))
            return s
        else: 
            print("bad operation name")

def generate_string_ends(op_string,bigrams):
    """Generates string based on operation string and bigrams markhov chain.
    Requires end markers in the bigrams
     and requires that the ops end exactly when the bigrams do"""
    s=["["] # we need a start string to calculate transitional probs for starting
    b=[]
    #print (op_string)
    #print (s)
    for op in op_string:
        #print (op)
        #print (s)
        if s[-1]==']':
            if op=='end':
                print (' '.join(s))
                return s
            else:
                print ("end marker before end operation")
                return
        if op=='mg':
            next_word=next_step(s[-1],bigrams)
            if next_word==None:
                print ("no transition")
                return

            s.append(next_word)
            b.append(next_word)
        elif op=='copy':
            s+=b
            b+=b
        elif op=='clear':
            b=[]
        elif op=='end':
            if s[-1]==']':
                print  (' '.join(s))
                return s
            else:
                print("end operation before end marker")
                return
        else: 
            print("bad operation name")
            return


            
##wrappers
def gen(bigrams,ops):
    return generate_string(generate_ops(ops)[0],bigrams)

def gen_ends(bigrams_ends,ops):
    return generate_string_ends(generate_ops(ops)[0],bigrams_ends)


def gen_corpus(bigrams,ops,n):
    """
    generates a corpus without end markers in the bigrams
    """
    i=0
    corpus=[]
    while i<n:
        corpus.append(gen(bigrams,ops))
    return corpus


def gen_corpus_ends(bigrams,ops,n):
    """
    generates a corpus with end markers in the bigrams
    """
    corpus=[]
    while len(corpus)<n:
        s = gen_ends(bigrams,ops)
        if s!=None:
            corpus.append(s)
    return corpus


######### PARSE ###########


def check_edge(e,q,fsm):
    """
    checks whether the current state q has an out-edge e. If so returns the state it goes to and the probability; otherwise returns None
    
    Arguments
    e   : we're checking to see if there's an edge with this label
    q   : the current state
    fsm : the fsm

    Returns 
    (q2,p): the destination and probability of the edge

    """
    possible_states=fsm[q]
    for s in possible_states:
        #print (s)
        if e in possible_states[s]:
            return (s,possible_states[s][e])
    

def possible_transitions(q,fsm):
    ts=[]
    possible_states=fsm[q]
    for s in possible_states:
        #print (s)
        for e in possible_states[s]:
            ts.append((s,e,possible_states[s][e]))
    return ts



def copy_and_apply_transition(state,t,fsm,bigrams,s,verbose=False):
    """ 
    Applies transition t in the given state,
    if this is actually possible given the string
    and buffer and bigram chain; if so, returns a copy of the state
    with the transition applied and the pointer 
    in the string advanced to the new location
    Also checks whether we just completed a parse.
    If this transition is not possible, return None.

    Arguments
    state   : aka an agenda task of the form (bigrams used,buffer,(states visited,operations used),index for where we are in the sentence)
    t       : transition (new state, operation)
    fsm     : the operations FSA
    bigrams : the bigram Markhov chain
    s : sentence
    verbose : for debugging

    Returns
    (new state, gram) where gram is true if this is a complete, valid parse
    """
    #if verbose: print ("copy and apply transition")
    (bis,buf,(qs,ops),k,p)=state
    new_bis,new_buf,new_qs,new_ops=bis[:],buf[:],qs[:],ops[:] # make a copy

    (next_state,op,p_new)=t
    n=len(s)
    gram=False # this will be true if we end at the final state

    if op=='clear':
        if verbose: print (" Clear")
        new_buf=[] # clear the buffer

    elif op=='copy':
        if verbose: print (" Copy")
        copy_end = k+len(buf)
        if copy_end>n:
            if verbose: print ("no room for a copy")
            return  # no room for a copy here
        if verbose: print ("buffer ",buf)
        if verbose: print ("copy? ", s[k : copy_end])
        if buf==s[k : copy_end]: # if this is a copy            
            new_buf+=new_buf # copy the buffer
            k=copy_end # move the pointer to the end of the copy
        else: 
            if verbose: print ("not a copy")
            return 

    elif op=='mg':
        if verbose: 
            print (" Merge")
            
        if k>n-1:
            if verbose: print ("no room to Merge")
            return # if we don't have room for more merging

        if verbose: print (bis[-1],s[k])

        if s[k] in bigrams[bis[-1]]: # if the bigram is allowed
            new_bis.append(s[k]) # add the new bigram
            new_buf.append(s[k])
            p+=bigrams[bis[-1]][s[k]]
            k+=1 # move over one word
        else: 
            if verbose: print ('illegal bigram %s %s'%(s[k],bis[-1]))
            return 

    elif op=='end' and k==n:
        if verbose: print (" End") # might want to add a ']' here?
        gram=True # this is grammatical!
 
    #if this wasn't going to work we'd've bailed out by now, 
    #so apply the transition to the FSM record
    new_qs.append(next_state) # add the new state
    new_ops.append(op) # add the operation

    if verbose: print ("new state: ",(new_bis,new_buf,(new_qs,new_ops),k,p+p_new),gram)
    
    return ((new_bis,new_buf,(new_qs,new_ops),k,p+p_new),gram)





def parse(s,bigrams,fsm,start='S',verbose=False):
    """
    Parses a string
    Uses an agenda and a list of complete parses.
    We initialise the agenda with a start state,  [ (['['],[],(['S'],[]),1) ]
    An agenda item is a list of 4 elements:
    the string so far, the buffer, the route so far (states,ops), and the index we're at in the sentence, k
    We add onto an agenda item until it is a complete parse, in which case we move it to the output, parses
        
    Arguments
    s       : the string to be parsed -- string
    bigrams : bigram markhov chain
    fsm     : operations FSM
    verbose : for debugging

    Returns
    list of complete parses
    """

    agenda = [ (['['],[],([start],[]),0,0.) ] # initialise with both start categories

    complete = [] # for the complete parses

    tries=0

    s=s.split(' ')
    
    while len(agenda)>0:
        task = agenda[0] # take the first agenda item
        (bis,buf,(qs,ops),k,p)=task # extract the current task
        if verbose: print ("\nprefix: %s"%(' '.join(s[:k])))

        for t in possible_transitions(qs[-1],fsm): # loop through possible transitions in FSA
            if verbose: print ("\ntransition: ",t)
            tries+=1
            result = copy_and_apply_transition(task,t,fsm,bigrams,s,verbose) # try to apply each one
            if result!=None: # every time we fail to apply a transition, result=None
                (newtask,gram)=result
                if gram: # if it's a complete grammatical sentence
                    if verbose: print ("done this task")
                    complete.append(newtask) # add it to the complete ones
                else: # otherwise, add the new task (task with t applied) back to the agenda
                    agenda.append(newtask) 

        del agenda[0] # remove the task we just completed

    if verbose: 
        print ("\nParses")
        for p in complete:
            print (parse2string(p))
        print ("\nAttempts to transition: %i"%tries)
        print ("Number of parses: %i"%len(complete))
    return complete









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




### WRAPPERS #####

def string2print(s,bigrams,fsm,start='S'):
    """
    prints to terminal trees of the parses of the string

    Arguments
    s         : sentence in list form
    bigrams   : bigram markhov chain
    fsm       : operation fsm
    start     : start category
    """
    prob=log0
    parses=parse(s,bigrams,fsm,start)
    for i,p in enumerate(parses):
        print ("\nprob of parse %i: %.4f"%(i,p[-1]))
        print ("Bigrams: %s"%(' '.join(p[0])))
        print ("Operations: %s"%(' '.join(p[2][1])))
        print ("Operation states: %s"%(' '.join(p[2][0])))

                #print (parse2tree(p))
        prob=log_add(prob,p[-1])


    print ("Number of parses: %i"%len(parses))
    print ("Prob of sentence: %.5f"%prob)

    


def string2pics(s,bigrams,fsm,file_type,start='S'):
    """
    prints to file trees of the parses of the string

    Arguments
    s         : sentence in list form
    bigrams   : bigram markhov chain
    fsm       : operation fsm
    file_type : string: 'pdf' or 'png' for type of file you want graphviz to make
    start     : start category
    """
    prob=log0
    parses=parse(s,bigrams,fsm,start)
    for i,p in enumerate(parses):
        parse2pic(p,"parse_%i"%i,file_type)
        print ("prob of parse %i: %.4f"%(i,p[-1]))
        prob=log_add(prob,p[-1])


    print ("Number of parses: %i"%len(parses))
    print ("Prob of sentence: %.5f"%prob)


###### CALCUATING THINGS ABOUT THE CORPUS #########

def ambiguity(corpus,bigrams,fsm,start='S'):
    total=0
    for s in corpus:
        parses=parse(s,bigrams,fsm,start)
        total+=len(parses)

    print ("%i parses for %i sentences = %.3f parses per sentence"%(total,len(corpus),float(total)/len(corpus)))
