#from importlib import reload #python3 only

import markhov
import numpy as np




bigrams = {'a':{']':0.25,'a':0.25,'b':0.5},
           'b':{']':0.25,'b':0.25,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }


# markhov chain of operations
# when you go to Merge, you also make a move in the bigram FSM
# when you go to copy, you make a copy of the buffer and append it to the string and buffer
# when you go to clear, you clear the buffer
ops = {'mg':{'mg':0.8,'copy':0.1,'clear':0.1},
         'copy':{'mg':0.3,'copy':0.2,'clear':0.5},
         'clear':{'mg':1.}
     }


# log em up
for a in bigrams:
    for b in bigrams[a]:
        bigrams[a][b]=np.log(bigrams[a][b])

for a in ops:
    for b in ops[a]:
        ops[a][b]=np.log(ops[a][b])



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

