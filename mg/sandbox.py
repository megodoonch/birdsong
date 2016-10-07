import cky
import numpy as np
from importlib import reload

def make_sent(string):
    s=string.split(" ")
    return ["["]+s+["]"]
    

s = "[ a b c b c a a ]"
s=s.split(' ')


s2 = "a d e"
s2=s2.split(" ")
sentence=s2
n=len(sentence)


s=s.split(' ')

sentence = s

n=len(sentence)

trans = []
for i in range(n-1):
    trans.append((s[i],s[i+1]))
list(set(trans))

def count_ngrams(tr,a):
    tot = 0
    for b in tr[a]:
        tot+=tr[a][b]
    for b in tr[a]:
        tr[a][b]=np.log(tr[a][b]/float(tot))
    return tr[a]


def make_transitions(sents):
    transitions = {}
    for s in sents:
        for i in range(len(s)-1):
            if s[i] in transitions:
                transitions[s[i]][s[i+1]] = transitions[s[i]].get(s[i+1],0) + 1
            else:
                transitions[s[i]] = transitions.get(s[i],{s[i+1]:1})
    for a in transitions:
        count_ngrams(transitions,a)
    return transitions



transitions = {}
for i in range(len(s)-1):
    if s[i] in transitions:
        transitions[s[i]][s[i+1]] = transitions[s[i]].get(s[i+1],0) + 1
    else:
        transitions[s[i]] = transitions.get(s[i],{s[i+1]:1})

for a in transitions:
    count_ngrams(transitions,a)

trans.append(("a","a"))


for i,tree in enumerate(trees):
    tree_to_pdf(tree,"parse_%i.pdf"%i)
