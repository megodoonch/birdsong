"""
A CKY parser for sets of operation strings 

"""
import numpy as np
import random


from markhov import log0, log_add, ops_log, ops, bi_ops

ops_psg = {'S':{('MG','NotCL'):1.},
           'NotCL':{('MG','NotCL'):0.25,
                    ('COPY','NotCL'):0.25,
                    ('CLEAR','CLS'):0.25,
                    'end':0.25},
           'CLS':{('MG','CL'):1.},
           'CL':{('MG','CL'):0.5,
                 ('COPY','NotCL'):0.5},
           'MG':{'mg':1.},
           'COPY':{'copy':1.},
           'CLEAR':{'clear':1.}
       }


def cat(word,psg=ops_psg):
    for lhs in psg:
        for rhs in psg[lhs]:
            if rhs == word:
                return ((lhs,rhs),psg[lhs][rhs])
    print ("No matching category for %s"%word)


def add_cell_element(cell,e):
    print("adding element ",e)
    (cats,p)=e
    cell[cats[0]]=cell.get(cats[0],{'bp':[],'p':1.})
    if cats[:1] not in cell[cats[0]]['bp']:
        cell[cats[0]]['bp'].append(cats[1:]) # add rhs to back pointers
    cell[cats[0]]['p']+=p


def chart2string(chart):
    s=""
    for i in chart:
        s+="\n%i:\n  lex:\n"%i
        for lhs in chart[i]['lex']:
            s+="   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['lex'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="     p=%.4f"%chart[i]['lex'][lhs]['p']
        s+="\n%i:\n  phrases:\n"%i
        for lhs in chart[i]['phrase']:
            s+="   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['phrase'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="     p=%.4f"%chart[i]['phrase'][lhs]['p']

    return s


s1="mg clear mg copy end".split(' ')
s2="mg mg mg end".split(' ')

parse=[s1,s2]

def initialise(parse):

    chart = {}
    for s in parse:
        n=len(s)-1
        for i in range(n):
            chart[i]=chart.get(i,{'lex':{}, 'phrase':{}})
            add_cell_element(chart[i]['lex'],cat(s[i])) # add the new item
        chart[n]=chart.get(n,{'lex':{}, 'phrase':{}})
        add_cell_element(chart[n]['phrase'],cat(s[n])) # add the new item
    return chart


n=len(chart)

i=n-2
j=n-1

new_cells = []

for left in chart[i]['lex']['bp']:
    for right in chart[j]['phrase']['bp']:
        d1=left[0]
        d2=right[0]
        for lhs in ops_psg:
            for rhs in ops_psg[lhs]:
                if rhs==(d1,d2):
                    new_cells.append(((lhs,rhs),ops_psg[lhs][rhs]))
                    

for cell in new_cells:
    chart[j]['phrase']=add_cell_element(chart[j]['phrase'],cell)
