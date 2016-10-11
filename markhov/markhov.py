"""
Two Marknov chains, one with bigrams and the other with merge, copy, and clear

Build two parallel strings, string and buffer. Copy and clear buffer.
"""



bigrams = {'a':{']':0.25,'a':0.25,'b':0.5},
           'b':{']':0.25,'b':0.25,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }


# markhov chain of operations
# when you go to Merge, you also make a move in the bigram FSM
# when you go to copy, you make a copy of the buffer and append it to the string and buffer
# when you do to clear, you clear the buffer
ops = {'mg':{'mg':0.8,'copy':0.1,'clear':0.1},
         'copy':{'mg':0.3,'copy':0.2,'clear':0.5},
         'clear':{'mg':1.}
     }


start_state=(([],[],1.),('[','mg'))

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
    return (string,buf,p*p_copy)

def clear(string, buf,p, p_clear):
    """Clear the buffer, multiplies in prob of clearing.
    Arguments:
    string : string list. The string you're actually generating
    buf    : string list. The buffer you might copy
    p      : float. The current probability of the string
    p_clear : float. The probability of transitioning to Clear 
               from whereever we currently are in the ops FSM
    """

    return (string, [], p*p_clear)

def merge(string,buf,p,p_merge,last_word,next_word):
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

    string.append(next_word)
    buf.append(next_word)
    return (string, buf, p*p_merge*bigrams[last_word][next_word])


def move(current,new_state,bigrams,ops):
    """
    Applies the move we want given in new_state to current output and states
    Arguments:
    current    : pair of current output and current state
                 current output: triple of (string,buffer,probability)
                 states: pair of (state in bigrams, state in ops)
    """
    ( (string,buf,p),
    (last_word,last_op) ) = current
    
    if new_state=='copy':
        assert new_state in ops[last_op].keys() , "you can't copy after %s"%last_op
        out= ( copy(string,buf,p,ops[last_op]['copy']), (last_word,'copy') )
    elif new_state=='clear':
        #assert not cleared , "You can't clear again already"
        assert new_state in ops[last_op].keys() , "you can't clear after %s"%last_op
        out= ( clear(string,buf,p,ops[last_op]['clear']), (last_word,'clear') )
    else:
        assert (len(new_state)==2 and new_state[0]=='mg') , "bad operation name"
        assert new_state[0] in ops[last_op].keys() , "you can't merge after %s"%last_op
        (op,next_word)=new_state
        out= ( merge(string,buf,p,ops[last_op]['mg'],last_word,next_word) , (next_word,'mg') )
    print("string: %s"%(' '.join(out[0][0])))
    print("buffer: %s"%(' '.join(out[0][1])))
    print("p: %f"%out[0][2])
    return out
    
