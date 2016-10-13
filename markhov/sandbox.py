from importlib import reload #python3 only

import markhov
import numpy as np
import random




# bigrams = {'a':{']':0.25,'a':0.25,'b':0.5},
#            'b':{']':0.25,'b':0.25,'a':0.5},
#            '':{'a':0.5,'b':0.5}
#        }

def ops_log(ops):
    for a in ops:
        for b in ops[a]:
            for w in ops[a][b]:
                ops[a][b][w]=np.log(ops[a][b][w])
    return ops

def bis_log(bigrams):
    for a in bigrams:
        for b in bigrams[a]:
            bigrams[a][b]=np.log(bigrams[a][b])
    return bigrams


bigrams = {'a':{'a':0.5,'b':0.5},
           'b':{'b':0.5,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }

bigrams=bis_log(bigrams)

bigrams_ends = {'a':{'a':0.5,'b':0.25,']':0.25},
           'b':{'b':0.5,'a':0.25,']':0.25},
           '[':{'a':0.5,'b':0.5}
       }

bigrams_ends=bis_log(bigrams_ends)


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

ops=ops_log(ops)

bi_ops = {'S':{'S':{'mg':0.5},
               'F':{'end':0.5}
           },
          'F':{}
      }

bi_ops=ops_log(bi_ops)





