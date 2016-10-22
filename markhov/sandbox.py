from importlib import reload #python3 only

import markhov
import numpy as np
import random



a="a"

aaa = ['a','a a','a a a']

ops = {'S':{'NotCL':['mg']}, # from start we have to merge
       'NotCL':{'NotCL':['mg','copy'], # this state is the state in which the last "special" operation was *not* Clear. Either we've done none or the last was copy. From here we can do everything including end
               'CLEAR_S':['clear'], # go here to clear the buffer
               'F':['end'] # go here to end
           },
       'CLEAR_S':{'CLEAR':['mg']}, # this is where we've just cleared. Buffer is empty so you can only Merge
       'CLEAR':{'CLEAR':['mg'], # the last special op was Clear so we can Copy or Merge.
                'NotCL':['copy'] # if we Copy, the last special op was Copy so go to NotCL
            },
       'F':{} #final state
   }


trans = {'a':['a','b'],
           'b':['b','a'],
           '[':['a','b']
       }

    
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




trans_probs = {'a':{'a':0.5,'b':0.5},
           'b':{'b':0.5,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }

trans_probs=bis_log(trans_probs)

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
ops_probs = {'S':{'NotCL':{'mg':1.}}, # from start we have to merge
       'NotCL':{'NotCL':{'mg':0.3,'copy':0.1}, # this state is the state in which the last "special" operation was *not* Clear. Either we've done none or the last was copy. From here we can do everything including end
               'CLEAR_S':{'clear':0.1}, # go here to clear the buffer
               'F':{'end':0.5} # go here to end
           },
       'CLEAR_S':{'CLEAR':{'mg':1.}}, # this is where we've just cleared. Buffer is empty so you can only Merge
       'CLEAR':{'CLEAR':{'mg':0.5}, # the last special op was Clear so we can Copy or Merge.
                'NotCL':{'copy':0.5} # if we Copy, the last special op was Copy so go to NotCL
            },
       'F':{} #final state
   }

ops_probs=ops_log(ops_probs)

bi_ops = {'S':{'S':{'mg':0.5},
               'F':{'end':0.5}
           },
          'F':{}
      }

bi_ops=ops_log(bi_ops)


s1 = "mg mg end".split(' ')
s2 = "mg copy end".split(' ')
bis1 = "[ a a".split(' ')
bis2= "[ a".split(' ')


s1="mg clear mg copy mg end".split(' ')
s2="mg mg mg mg end".split(' ')
bis1="[ a b a".split(' ')
bis2="[ a b b a".split(' ')

parses = [(s1,bis1),(s2,bis2)]

for lhs in ops_psg:
    for (rhs,p) in ops_psg[lhs]:
        print lhs,rhs,get_c_phi((lhs,rhs),parses,ops_psg,bigrams)



rule=('S',['MG','NotCL'])

mg = ('MG',['mg'])

copy= ('COPY',['copy'])





f=open('../corpus/cath8.txt','r')
corpus = f.readlines()
f.close()
corpus = [line.rstrip('\n') for line in corpus]
