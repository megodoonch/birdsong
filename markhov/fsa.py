"""
The finite state automata for operations

"""

import numpy as np
import random


def ops_log(ops):
    """Makes probs logarithmic"""
    for a in ops:
        for b in ops[a]:
            for w in ops[a][b]:
                ops[a][b][w]=np.log(ops[a][b][w])
    return ops



def ops_random(ops):
    """
    gives operations random probabilities that sum to 1 for each lhs

    Arguments
    ops : operation FSA as a dict of dicts of dicts

    Returns
    ops with probs replaced with random (log) probs
    """
    for a in ops:
        n=0
        for b in ops[a]:
            n+=len(ops[a][b])
        r = [random.random() for i in range(n)]
        s = sum(r)
        r = [ i/s for i in r ]
        i=0
        for b in ops[a]:
            for e in ops[a][b]:
                ops[a][b][e] = np.log(r[i])
    return ops
                


ops = {'S':{'NotCL':{'mg':1.}}, # from start we have to merge
       'NotCL':{'NotCL':{'mg':0.25,'copy':0.25}, # this state is the state in which the last "special" operation was *not* Clear. Either we've done none or the last was copy. From here we can do everything including end
                'CLEAR_S':{'clear':0.25}, # go here to clear the buffer
               'F':{'end':0.25} # go here to end
           },
       'CLEAR_S':{'CLEAR':{'mg':1.}}, # this is where we've just cleared. Buffer is empty so you can only Merge
       'CLEAR':{'CLEAR':{'mg':0.5}, # the last special op was Clear so we can Copy or Merge.
                'NotCL':{'copy':0.5} # if we Copy, the last special op was Copy so go to COPY
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
