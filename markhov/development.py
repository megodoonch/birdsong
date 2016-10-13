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
