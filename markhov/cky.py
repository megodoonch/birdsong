"""
A CKY parser for sets of operation strings 

"""
import numpy as np
import random


from markhov import log0, log_add, ops_log, ops, bi_ops


ops_psg = {'S':[(['MG','NotCL'],1.)],
           'NotCL':[(['MG','NotCL'],0.25),
                    (['COPY','NotCL'],0.25),
                    (['CLEAR','CLS'],0.25),
                    (['end'],0.25)],
           'CLS':[(['MG','CL'],1.)],
           'CL':[(['MG','CL'],0.5),
                 (['COPY','NotCL'],0.5)],
           'MG':[(['mg'],1.)],
           'COPY':[(['copy'],1.)],
           'CLEAR':[(['clear'],1.)]
       }


def cat(word,psg=ops_psg):
    """
    Gets the category of a word from the phrase structure grammar
    """
    for lhs in psg:
        for (rhs,p) in psg[lhs]:
            if len(rhs)==1 and rhs[0] == word:
                return ((lhs,rhs),p)
    print ("No matching category for %s"%word)


def add_cell_element(cell,e):
    """
    Adds a new element to a CKY cell.

    Arguments:
    cell : lex: { LHS: {backpointers , prob}, phrase : same }
            lex is the cell from the diagonal, phrase is the cell from the last column
    e    : ((lhs,rhs),p)    
    """
    print("adding element ",e)
    ((lhs,rhs),p)=e
    cell[lhs]=cell.get(lhs,{'bp':[],'p':0.}) # make a cell element if necesary for this LHS
    print("cell:",cell)
    if rhs not in cell[lhs]['bp']: # add the RHS if not already there
        print("RHS",rhs)
        cell[lhs]['bp'].append(rhs) # add rhs to back pointers
    cell[lhs]['p']+=p # add in the probability


def chart2string(chart):
    """
    Makes a string from a chart for printing
    """
    s=""
    for i in chart:
        s+="\n%i:\n  lex:"%i
        for lhs in chart[i]['lex']:
            s+="\n   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['lex'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="     p=%.4f"%chart[i]['lex'][lhs]['p']
        s+="\n  phrases:"
        for lhs in chart[i]['phrase']:
            s+="\n   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['phrase'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="    p=%.4f"%chart[i]['phrase'][lhs]['p']

    return s


s1="mg clear mg copy end".split(' ')
s2="mg mg mg end".split(' ')

parse=[s1,s2]

def initialise(parse):
    """
    Initialises the CKY chart based on a list of parse strings
    """
    chart = {}
    for s in parse:
        n=len(s)-1 # the last element is special
        for i in range(n): # go through the sentence
            chart[i]=chart.get(i,{'lex':{}, 'phrase':{}}) # make a row if nec
            add_cell_element(chart[i]['lex'],cat(s[i])) # add the new item to the diagonal
        chart[n]=chart.get(n,{'lex':{}, 'phrase':{}}) # put the last element, 
        #which we hope is "end", 
        #in the last column of its usual row. 
        #This allows us to parse different length sentences in the same chart
        print(cat(s[n]))
        add_cell_element(chart[n]['phrase'],cat(s[n])) # add the new item
    return chart


def cky(parse):
    """
    Creates and fills a (simplified for regular grammars) CKY chart for a set of parse strings

    Arguments
    parse : list of strings of operations

    Returns
    filled CKY "chart" 
    which is actually a dictionary with keys standing for the rows, 
    which are themselves dicts of 
    lex (standing for the main diagonal) and 
    phrase (standing for the last column)

    cells are also dicts, with keys categories and values dicts
    of backpointers (bp) and probabilities (p)
    """
    chart=initialise(parse) # add the words

    n=len(chart)

    for j in list(reversed(range(1,n))): # run upwards through the rows
        # j is the row for the right sister
        i=j-1 # you always look in the lex cell of the row above for your sister
        print ((i,j))
        new_cell_elements = [] # initialise a list of new cell elements to add to the chart
 
        for left in chart[i]['lex']: # try all the left sisters
            for right in chart[j]['phrase']: # try all the right sisters
                print((left,right))
                for lhs in ops_psg: # check in the grammar
                    for (rhs,p) in ops_psg[lhs]: 
                        if rhs==[left,right]: # if this is the rule we want
                            p=p*chart[i]['lex'][left]['p']*chart[j]['phrase'][right]['p'] # calculate new prob
                            new_cell_elements.append(((lhs,rhs),p)) # new cell element


        for e in new_cell_elements: 
            # add the new elements to the phrase cell at the end of the left sister's row
            add_cell_element(chart[i]['phrase'],e)

    print(chart2string(chart))
    return chart
