"""
A CKY parser for  operation strings 

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


bigrams = {'a':{'a':0.5,'b':0.5},
           'b':{'b':0.5,'a':0.5},
           '[':{'a':0.5,'b':0.5}
       }


def merge_p(bi,p,bigrams):
    """
    Calculates the merge probability by combining the probs of the bigram of words and the prob of merge. 

    Arguments
    bi : bigram
    p  : p(MG->mg) from grammar (should be 1)
    
    Returns
    combined probability of merge op and bigram
    """
    (w1,w2)=bi
    return bigrams[w1][w2]*p


def cat(word,psg,bi=None,bigrams=None):
    """
    Gets the category of a word from the phrase structure grammar
    """
    for lhs in psg:
        for (rhs,p) in psg[lhs]:
            if len(rhs)==1 and rhs[0] == word:
                if word=='mg' and bi !=None:
                    p=merge_p(bi,p,bigrams) # add in bigram prob if given
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
            s+="     p=%.9f"%chart[i]['lex'][lhs]['p']
        s+="\n  phrases:"
        for lhs in chart[i]['phrase']:
            s+="\n   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['phrase'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="    p=%.9f"%chart[i]['phrase'][lhs]['p']

    return s


def chart2string_beta(chart):
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
            s+="     p=%.9f"%chart[i]['lex'][lhs]['p']
            s+="     beta=%.9f"%chart[i]['lex'][lhs].get('beta',0.)

        s+="\n  phrases:"
        for lhs in chart[i]['phrase']:
            s+="\n   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['phrase'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="    p=%.9f"%chart[i]['phrase'][lhs]['p']
            s+="    beta=%.9f"%chart[i]['phrase'][lhs].get('beta',0.)

    return s


s1="mg clear mg copy mg end".split(' ')
s2="mg mg mg mg end".split(' ')

parse=[s1,s2]



def initialise(s,psg):
    """
    Initialises the CKY chart based on a parse string
    """
    chart = {}
    n=len(s)-1 # the last element is special
    for i in range(n): # go through the sentence
        chart[i]=chart.get(i,{'lex':{}, 'phrase':{}}) # make a row if nec
        add_cell_element(chart[i]['lex'],cat(s[i],psg)) # add the new item to the diagonal
    chart[n]=chart.get(n,{'lex':{}, 'phrase':{}}) # put the last element, 
    #which we hope is "end", 
    #in the last column of its usual row. 
    print(cat(s[n]))
    add_cell_element(chart[n]['phrase'],cat(s[n],psg)) # add the new item
    return chart


def cky(s,psg):
    """
    Creates and fills a (simplified for regular grammars) CKY chart for a set of parse strings

    Arguments
    s : string of operations

    Returns
    filled CKY "chart" 
    which is actually a dictionary with keys standing for the rows, 
    which are themselves dicts of 
    lex (standing for the main diagonal) and 
    phrase (standing for the last column)

    cells are also dicts, with keys categories and values dicts
    of backpointers (bp) and probabilities (p)
    """
    chart=initialise(s,psg) # add the words

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


def parse(s,psg,bis=None,bigram_chain=None):
    """
    
    """
    if bis !=None: bis = list(reversed(bis)) # we're going up through the chart so we go backwards through the bigram string

    def cat(word,bi=None):
        """
        Gets the category of a word from the phrase structure grammar
        """
        for lhs in psg:
            for (rhs,p) in psg[lhs]:
                if len(rhs)==1 and rhs[0] == word:
                    if word=='mg' and bi !=None:
                        p=merge_p(bi,p,bigram_chain) # add in bigram prob if given
                    return ((lhs,rhs),p)
        print ("No matching category for %s"%word)

    #initialise the chart
    chart = {}
    n=len(s)-1 # the last element is special
    k=0 # counter to move us through the bigrams. We only move when we Merge.
    for i in range(n): # go through the sentence
        bi=None # initialise to None
        chart[i]=chart.get(i,{'lex':{}, 'phrase':{}}) # make a row if nec
        if bis!=None and s[i]=='mg':
            bi=(bis[k+1],bis[k]) # bigrams are backwards
            k+=1 # move through the bigram string
        add_cell_element(chart[i]['lex'],cat(s[i],bi)) # add the new item to the diagonal
    chart[n]=chart.get(n,{'lex':{}, 'phrase':{}}) # put the last element, 
    #which we hope is "end", 
    #in the last column of its usual row. 
    #print(cat(s[n]))
    add_cell_element(chart[n]['phrase'],cat(s[n])) # add the new item

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


def outside(chart,psg,start='S'):

    # initialise all betas to 0
    for i in chart:
        for lhs in chart[i]['phrase']:
            chart[i]['phrase'][lhs]['beta']=0.
        for lhs in chart[i]['lex']:
            chart[i]['lex'][lhs]['beta']=0.

    chart[0]['phrase'][start]['beta']=1. # initialise top right start to 1
    c=start
    def beta_cell(i,c): 
        """row i, category c"""
        print("\nfor",i,c)
        for bp in chart[i]['phrase'][c]['bp']:
            if len(bp)==2: # phrasal rule
                left,right=bp[0],bp[1]
                left_p = chart[i]['lex'][left]['p']
                right_p = chart[i+1]['phrase'][right]['p']
                print("right: %s p: %.5f\nleft: %s p: %.5f"%(right,right_p,left,left_p))
                rules = psg[c]
                rule_p=0.
                for (rhs,p) in rules:
                    if rhs == [left,right]: # if it's the right rule
                        rule_p=p # get this prob
                print("rule p: %.5f"%rule_p)
                beta = chart[i]['phrase'][c]['beta']# beta of mother
                print("beta: %.5f"%beta)
                #for each daughter, calculate the outside prob = p(rule) x beta(mother) x p(sister)
                chart[i]['lex'][left]['beta']+=rule_p*beta*right_p # put the outside prob in the chart
                chart[i+1]['phrase'][right]['beta']+=rule_p*beta*left_p # put the outside prob in the chart
                

    for i in range(len(chart)):
        for c in chart[i]['phrase']:
            beta_cell(i,c)

    print(chart2string_beta(chart))
    return chart



