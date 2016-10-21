"""Some old functions"""

def copy(string, buf, p, p_copy):
    """Copies the buffer, appends it to both string and buffer, multiplies in prob of copying.
    Arguments:
    string : string list. The string you're actually generating
    buf    : string list. The buffer you might copy
    p      : float. The current probability of the string
    p_copy : float. The probability of transitioning to Copy 
               from whereever we currently are in the ops FSM
    """
    string+=buf
    buf+=buf
    return (string,buf,p+p_copy)

def clear(string, buf,p, p_clear):
    """Clear the buffer, multiplies in prob of clearing.
    Arguments:
    string : string list. The string you're actually generating
    buf    : string list. The buffer you might copy
    p      : float. The current probability of the string
    p_clear : float. The probability of transitioning to Clear 
               from whereever we currently are in the ops FSM
    """

    return (string, [], p+p_clear)

def merge(string,buf,p,p_merge,last_word,next_word,bigrams):
    """Adds a new word to the end of the string and buffer, multiplies in prob of merging.
    Arguments:
    string    : string list. The string you're actually generating
    buf       : string list. The buffer you might copy
    p         : float. The current probability of the string
    p_merge   : float. The probability of transitioning to Merge
               from whereever we currently are in the ops FSM
    last_word : the current state in the bigrams FSM
    next_word : the word we're trying to append
    """
    string=string[:] #why do I have to do this???
    buf=buf[:]
    string.append(next_word)
    buf.append(next_word)
    return (string, buf, p+p_merge+bigrams[last_word][next_word])



##### applying the move #######

def move(current,new_state,bigrams,ops):
    """
    Applies the move we want given in new_state to current output and states
    Arguments:
    current    : pair of current output and current state
                 current output: triple of (string,buffer,probability)
                 states: pair of (state in bigrams, state in ops)
    """
    ( (string,buf,p),
    (last_word,last_op,cleared) ) = current
    
    if new_state=='copy':
        assert new_state in ops[last_op].keys() , "you can't copy after %s"%last_op
        assert len(buf)>0 , "you can't copy an empty buffer"
        out= ( copy(string,buf,p,ops[last_op]['copy']), (last_word,'copy',False) )
    elif new_state=='clear':
        #assert not cleared , "You can't clear again already"
        assert len(buf)>0 , "you can't clear an empty buffer"
        assert cleared==False , "you can't clear again until you copy"
        assert new_state in ops[last_op].keys() , "you can't clear after %s"%last_op
        out= ( clear(string,buf,p,ops[last_op]['clear']), (last_word,'clear',True) )
    else:
        assert (len(new_state)==2 and new_state[0]=='mg') , "bad operation name"
        assert new_state[0] in ops[last_op].keys() , "you can't merge after %s"%last_op
        (op,next_word)=new_state
        out= ( merge(string,buf,p,ops[last_op]['mg'],last_word,next_word,bigrams) , (next_word,'mg',cleared) )
    print("string: %s"%(' '.join(out[0][0])))
    print("buffer: %s"%(' '.join(out[0][1])))
    print("p: %f\n"%out[0][2])
    return out
    

def generate(bigrams,operations,start=START_STATE):
    """
    Generates a sentence according to the probabilities of the operations and bigrams
    Arguments:
    start      : start state, usually ((['['],[],0.),('[','mg'))
    bigrams    : bigram markhov chain
    operations : operation markhov chain

    Output     : full state: ( (sentence,buffer,prob) (last word,last op) )
    """
    state=start[:] # start with the start state
    def step(st):        
        """randomly chooses a move
        Arguments:
        st : pair of current output and states
        """
        current_out,current_states=st 
        if len(current_out[1])==0: # if we have no buffer, we can only Merge
            next_op = 'mg'
        else:   
            next_op = next_step(current_states[1],operations) # randomly choose next operation
        print(next_op)
        next_state=next_op # if copy or clear, this is the next state
        if next_op=='mg': 
            next_wd=next_step(current_states[0],bigrams) # get the next word if Merging
            next_state=(next_op,next_wd) # need to include next word in merge state
        return move((current_out,current_states),next_state,bigrams,operations) # apply the move
        
    while state[0][0][-1] != ']': #stop when we get to the end symbol
        state=step(state)
    
    print ("sentence: %s\n"%(' '.join(state[0][0])))
    return state


def parse(s,bigrams):
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

    s = s.split(' ') # make the string into a list
    possible = len(s)>=2 and s[0]=='[' and s[-1]==']'
    if possible: # only sentence with at least [ and ] are going to be grammatical
        agenda = [ [['['],[],['mg'],False] ]  # initialise the agenda
    else:
        return (False,[])
    parses = [] # this will be our output
    while len(agenda)>0: # keep parsing as long as we have incomplete parses
        for parse in agenda:
            print("\nparse: %s"%parse)
            i=len(parse[0])
            if i==len(s): # move a complete parse over to the output
                parses.append(parse)
                agenda.remove(parse)
            else:
                if len(parse[1])>0: # if buffer not empty
                    ## try clearing the buffer
                    print ("\n Clear")
                    if not parse[3]:  #i if last special op not clear
                        new_parse=[[],[],[],[]] # this is to make copies of the lists inside the list
                        new_parse[0]=parse[0][:] # copy the string
                        new_parse[2]=parse[2][:] # copy the list of operations
                        new_parse[1]=[] # clear the buffer
                        new_parse[2].append('clear')
                        new_parse[3]=True # the last special op was Clear
                        print ("new parse: %s"%new_parse)
                        agenda.append(new_parse) # add this new parse to the agenda

                    #Try to Copy
                    print ("\n Copy")
                    print ("buffer ",parse[1])
                    print ("copy? ", s[i : i+len(parse[1])])
                    # if the buffer is the same as the next part of the sentence
                    if i+len(parse[1]) <= len(s) and parse[1]==s[i : i+len(parse[1])]: 
                        new_parse=[[],[],[],[]] # make a deep copy
                        new_parse[2]=parse[2][:] # copy the list of operations
                        new_parse[0]=s[:i+len(parse[1])] # skip ahead to end of copy in the string
                        new_parse[1]=parse[1][:]+parse[1][:] #add copy to buffer
                        new_parse[2].append('copy') # add operation to operation list 
                        new_parse[3]=False # the last special operation was not Clear
                        print ("new parse : %s"%new_parse)
                        agenda.append(new_parse) # add the new parse to the agenda

                # Try to Merge
                print ("\n Merge")
                if s[i] in bigrams[s[i-1]]:
                    parse[0].append(s[i]) #string
                    parse[1].append(s[i]) #buffer
                    parse[2].append('mg') #list of ops
                    print ("merged parse: %s"%parse)
                else:
                    return (False,parses) # if this isn't a legal transition, this sentence is not grammatical

    return (True,parses)



def probability(s,bigrams,ops):
    """
    Parses a string and give the total probability of the parses.

    Returns two different things: a total probability for all parses
    and a total probability only for parses that don't have a Clear that isn't followed by a Copy.
    
    Uses an agenda and a list of complete parses.
    We initialise the agenda with a start state,   [['['],[],['mg'],False,0.]
    An agenda item is a list of 5 elements:
    the string so far, the buffer, the operations, and whether the last "special" operation (copy or clear) is clear.
    The second-last element is a little hack to reduce the amount of pointless buffer-clearing.
    the last is the probability of the parse
    We add onto an agenda item until it is a complete parse, in which case we move it to the output, parses
    Whenever there is a choice of two valid moves, we copy the agenda item and try both
        
    Arguments:
    s       : the string to be parsed -- string
    bigrams : bigram markhov chain
    ops     : operations markhov chain (mg,copy,clear)
    """

    s = s.split(' ') # make the string into a list
    parses = ((log0,[]),(log0,[])) # these will be our outputs: (sum prob,parses). First is all parses, second excludes parses with pointless buffer-clearing

    possible = len(s)>=2 and s[0]=='[' and s[-1]==']'
    
    if possible: # only sentence with at least [ and ] are going to be grammatical
        agenda = [ [['['],[],['mg'],False,0.,1] ]  # initialise the agenda
    else:
        return parses
    while len(agenda)>0: # keep parsing as long as we have incomplete parses. We have a chance to bail if there's a bad transition
        for parse in agenda:
            print("\nparse: %s"%parse)
            i=parse[5]
            if i==len(s): # move a complete parse over to the output
                (all_parses,useful_parses)=parses 
                all_parses = (log_add(all_parses[0],parse[4]), all_parses[1]+[parse]) # add prob of this parse to total prob
                copy_clear = [op for op in parse[2] if op !='mg'] # get just the non-merge operations
                #useful=True
                if copy_clear==[] or copy_clear[-1]!='clear':
                # for i in range(len(copy_clear)): # look for an instance of clear not followed by a copy
                #     #print ("i=%i"%i)
                #     #print (copy_clear)
                #     if copy_clear[i]=='clear' and (i==len(copy_clear)-1 or copy_clear[i+1] !='copy') :
                #         useful=False # if you find one this isn't a useful parse
                #         break
                #if useful:
                    useful_parses = (log_add(useful_parses[0],parse[4]), useful_parses[1]+[parse])
                parses=(all_parses,useful_parses)
                agenda.remove(parse)
            else:
                if len(parse[1])>0: # if buffer not empty
                    ## try clearing the buffer
                    print ("\n Clear")
                    if not parse[3]:  #i if last special op not clear
                        new_parse=[[],[],[],[],[],[]] # this is to make copies of the lists inside the list
                        new_parse[0]=parse[0][:] # copy the string
                        new_parse[2]=parse[2][:] # copy the list of operations
                        new_parse[1]=[] # clear the buffer
                        new_parse[2].append('clear')
                        new_parse[3]=True # the last special op was Clear
                        new_parse[4]=parse[4]+ops[parse[2][-1]]['clear']
                        new_parse[5]=parse[5] # we haven't advanced in the string
                        print ("new parse: %s"%new_parse)
                        agenda.append(new_parse) # add this new parse to the agenda

                    #Try to Copy
                    print ("\n Copy")
                    print ("buffer ",parse[1])
                    print ("copy? ", s[i : i+len(parse[1])])
                    # if the buffer is the same as the next part of the sentence
                    if i+len(parse[1]) <= len(s) and parse[1]==s[i : i+len(parse[1])]: 
                        new_parse=[[],[],[],[],[],[]] # make a deep copy
                        new_parse[2]=parse[2][:] # copy the list of operations
                        new_parse[0]=parse[0][:] # we're really just keeping track of transitions
                        new_parse[1]=parse[1][:]+parse[1][:] #add copy to buffer
                        new_parse[2].append('copy') # add operation to operation list 
                        new_parse[3]=False # the last special operation was not Clear
                        last_op=parse[2][-1]
                        print (last_op)
                        new_parse[4]=parse[4]+ops[last_op]['copy']
                        new_parse[5]=parse[5]+len(parse[1]) # skip ahead in the string
                        print ("new parse : %s"%new_parse)
                        agenda.append(new_parse) # add the new parse to the agenda

                # Try to Merge
                print ("\n Merge")
                if s[i] in bigrams[s[i-1]]:
                    parse[0].append(s[i]) #string
                    parse[1].append(s[i]) #buffer
                    parse[2].append('mg') #list of ops
                    parse[4]=parse[4]+ops[parse[2][-1]]['mg']+bigrams[s[i-1]][s[i]]
                    parse[5]=parse[5]+1 # advance down the string 1 step
                    print ("merged parse: %s"%parse)
                else:
                    return parses # if this isn't a legal transition, this sentence is not grammatical

    return parses







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



def change_p_parse(parse,bigrams,fsa,start='S',end='F'):
    """
    calculate new probability of parse and add it back into the parse
    """
    (bis,route,_)=parse
    return(bis,route,p_parse(parse,bigrams,fsa,start,end))

    


def sum_p_corpus(parsed_corpus):
    return  log_sum([p for (s,parses,p) in parsed_corpus])






def update_automata_oldmeaghan(corpus,bigrams,fsa,verbose=False):
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

    ## TODO: can fsa just be the rules themselves, not their probabilities? i.e. a dict the same as fsa except for the very last level.

    ## TODO: this is really two separate functions that don't need each other's stuff

    #TODO smoothing for unused rules
    #new_fsa = copy.deepcopy(fsa) #copy the ops FSA
    new_fsa = {}
    for lhs in fsa: # go through the FSA
        if verbose: print (lhs)
        new_fsa[lhs]={}
        for rhs in new_fsa[lhs]:
            #print (rhs)
            new_fsa[lhs][rhs]={}
            for e in new_fsa[lhs][rhs]:
                #print (e)
                #print (lhs,e,rhs)
                tot_tc=log0
                tot_sc=log0

                for (s,parses,p_s) in corpus: # go through parsed corpus
                    #print (s)
                    # add up all the TCs/SCs for the parses of this sentence
                    for (_,_,p,sc,tc,_,_) in parses:
                        #print (tc)
                        #print (sc)
                        if (lhs,e,rhs) in tc: # if this parse has this rule
                            # add in counts, times p(parse)/p(sent)
                            tot_tc=log_add(tot_tc, p-p_s + log(tc[(lhs,e,rhs)]))
                            #print ("tot_tc: %f"%tot_tc)
                        if lhs in sc:
                            tot_sc=log_add(tot_sc, p-p_s + log(sc[lhs]))
                            #print ("tot_sc: %f"%tot_sc)

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

