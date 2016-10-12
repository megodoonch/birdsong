#from importlib import reload #python3 only

import markhov
import numpy as np
import random



# bigrams = {'a':{']':0.25,'a':0.25,'b':0.5},
#            'b':{']':0.25,'b':0.25,'a':0.5},
#            '':{'a':0.5,'b':0.5}
#        }


bigrams = {'a':{'a':0.5,'b':0.5},
           'b':{'b':0.5,'a':0.5},
           '':{'a':0.5,'b':0.5}
       }


# log em up
for a in bigrams:
    for b in bigrams[a]:
        bigrams[a][b]=np.log(bigrams[a][b])


# markhov chain of operations
# when you go to Merge, you also make a move in the bigram FSM
# when you go to copy, you make a copy of the buffer and append it to the string and buffer
# when you go to clear, you clear the buffer
# ops = {'mg':{'mg':0.8,'copy':0.1,'clear':0.1},
#          'copy':{'mg':0.3,'copy':0.2,'clear':0.5},
#          'clear':{'mg':1.}
#      }

# (state : state: transition: prob, final)
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

# logs
for a in ops:
    for b in ops[a]:
        for w in ops[a][b]:
            ops[a][b][w]=np.log(ops[a][b][w])


def step(state):
    """
    Chooses a next state in ops PFSM based on probabilities in machine
    """
    p=np.log(random.random()) # random prob for deciding next move
    # we'll look for an interval in the probs of out-arrows in which p falls. 
    current_p=markhov.log0 # this is the bottom of the first interval
    nexts=ops[state] # possible next states

    for st in nexts:
        #print(st)
        for op in nexts[st]:
            #print (op)
            p_next=nexts[st][op] # the prob of this arrow
            #print (p_next)
            #print ("is %0.5f < %0.5f < %0.5f?"%(current_p,p,p_next))
            if p>=current_p and p < markhov.log_add(p_next,current_p): # if the random prob falls between the last prob and this one, we've found our result
                #print (st,op,p)
                return (st,op,p) # this is it!
            else:
                current_p=markhov.log_add(p_next,current_p) #otherwise, move bottom of interval up



def generate_ops(ops):
    """
    Generates a string of operation based on ops PFSM
    """
    state='S'
    out=[]
    p=0.
    next_step=step(state)
    while next_step!=None:
        out.append(next_step[1])
        p=p+next_step[2]
        next_step=step(next_step[0])
    return out,p


def generate_string(op_string):
    """Generates string based on operation string and bigrams markhov chain"""
    s=[""] # we need a start string to calculate transitional probs for starting
    b=[]
    for op in op_string:
        if op=='mg':
            next_word=markhov.next_step(s[-1],bigrams)
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






# for a in ops:
#     for b in ops[a]:
#         ops[a][b]=np.log(ops[a][b])



markhov.generate(bigrams,ops)

s= "[ a a ]"

s="a b"

s="[ ]"

s="[ a a a a a ]"

s="[ a a a ]"


parses = markhov.parse(s,bigrams)

print(parses)


probs = markhov.probability(s,bigrams,ops)

print("prob all parses: %0.5f"%probs[0][0])

print("prob useful parses: %0.5f"%probs[1][0])

