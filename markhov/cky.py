"""
A CKY parser for  operation strings 

"""
import numpy as np
import random


from markhov import log0, log_add, ops_log, ops, bi_ops

#### AUTOMATON AS PHRASE STRUCTURE GRAMMAR ####

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


##### PRINTING ######    
    
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
            s+="     beta=%.9f"%chart[i]['lex'][lhs].get('beta',0.)

        s+="\n  phrases:"
        for lhs in chart[i]['phrase']:
            s+="\n   LHS: %s\n    Backpointers:\n"%lhs
            for bp in chart[i]['phrase'][lhs]['bp']:
                s+="      ->%s\n"%('.'.join(bp))
            s+="    p=%.9f"%chart[i]['phrase'][lhs]['p']
            s+="    beta=%.9f"%chart[i]['phrase'][lhs].get('beta',0.)

    return s

##### UTILITY FUNCTIONS ######
    
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




def add_cell_element(cell,e):
    """
    Adds a new element to a CKY cell.

    Arguments
    cell : lex: { LHS: {backpointers , prob}, phrase : same }
            lex is the cell from the diagonal, phrase is the cell from the last column
    e    : ((lhs,rhs),p)

    Returns
    None
    """
    #print("adding element ",e)
    ((lhs,rhs),p)=e
    cell[lhs]=cell.get(lhs,{'bp':[],'p':0.}) # make a cell element if necesary for this LHS
    #print("cell:",cell)
    if rhs not in cell[lhs]['bp']: # add the RHS if not already there
        #print("RHS",rhs)
        cell[lhs]['bp'].append(rhs) # add rhs to back pointers
    cell[lhs]['p']+=p # add in the probability


def get_p_parse(chart,start='S'):
    return chart[0]['phrase'][start]['p']

def get_p_rule(rule,psg):
    (lhs,rhs)=rule
    for (r,p) in psg[lhs]:
        if r==rhs:
            return p

def get_p_sent(charts,start="S"):
    p_sent = 0.
    for chart in charts:
        p_sent+=get_p_parse(chart,start)
    return p_sent



########## PARSER #############


def parse(s,psg,bis=None,bigram_chain=None):
    """
    CKY parser for regular phrase structure grammar

    Since the grammar is regular, we don't need to build the whole chart. Only the diagonal and the last column will be used. As such, we build a dict of rows.

    Each row's key is the number and value is a dict of two cells: lex (the diagonal is always lexical) and phrase (the last column).
    Each cell is a dict with keys the category built and value the additional info:
    bp is backpointers, which are just RHSs since we know the partition
    p is the (inside) probability of the LHS

    We initialise the lex cells on the diagonal, except the last one which goes in the phrase element since it'll be used as a right daughter. (that row's lex is empty.)
    Then we proceed upward through the rows, filling each end cell with the result of combining the lex cell on that row with the phrase cell in the row below it.

    Arguments
    s          : the string of operations
    psg        : the operations PSG
    bis        : the bigram string used to make this sentence. This is optional since I'm not entirely sure it's right to include this here. We use these to calculate the probability of MG->mg rules.
    bigram_chain : markhov chain of bigrams in case we include bigram probs
    """
    if bis !=None: bis = list(reversed(bis)) # we're going up through the chart so we go backwards through the bigram string

    def cat(word,bi=None):
        """
        Gets the category of a word from the phrase structure grammar
            Arguments
        word     : string from operations
        bi       : the bigram associated with an instance of Merge

        Returns
        ((lhs of rule, rhs of rule), prob of rule)
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
        #print ((i,j))
        new_cell_elements = [] # initialise a list of new cell elements to add to the chart
 
        for left in chart[i]['lex']: # try all the left sisters
            for right in chart[j]['phrase']: # try all the right sisters
                #print((left,right))
                for lhs in ops_psg: # check in the grammar
                    for (rhs,p) in ops_psg[lhs]: 
                        if rhs==[left,right]: # if this is the rule we want
                            p=p*chart[i]['lex'][left]['p']*chart[j]['phrase'][right]['p'] # calculate new prob
                            new_cell_elements.append(((lhs,rhs),p)) # new cell element


        for e in new_cell_elements: 
            # add the new elements to the phrase cell at the end of the left sister's row
            add_cell_element(chart[i]['phrase'],e)

    #print(chart2string(chart))
    return chart


############## INSIDE/OUTSIDE ##################

def outside(chart,psg,start_beta=1.,start='S'):
    """
    Calculates the outside probabilities of the CKY entries and adds them into the chart

    I have the starting prob for the top right equal to the proportion of the sentence probability that the parse contributes. We might not need this: We might be able to multiply this in later, when we do the expected counts.

    Arguments
    chart     : filled CKY chart
    psg       : operations PSG
    start_beta: starting prob for the top right S. Usually this is 1, but we have multiple parses.
    start     : start category

    Returns
    CKY chart with outside probs added to cell entries under key 'beta'
    
    """

    
    # initialise all betas to 0
    for i in chart:
        for lhs in chart[i]['phrase']:
            chart[i]['phrase'][lhs]['beta']=0.
        for lhs in chart[i]['lex']:
            chart[i]['lex'][lhs]['beta']=0.

    chart[0]['phrase'][start]['beta']=start_beta # initialise top right start to proportion of sentence prob this parse represents
    c=start
    def beta_cell(i,c): 
        """row i, category c"""
        #print("\nfor",i,c)
        for bp in chart[i]['phrase'][c]['bp']:
            if len(bp)==2: # phrasal rule
                left,right=bp[0],bp[1]
                left_p = chart[i]['lex'][left]['p']
                right_p = chart[i+1]['phrase'][right]['p']
                #print("right: %s p: %.5f\nleft: %s p: %.5f"%(right,right_p,left,left_p))
                rules = psg[c] #rule set for this LHS
                rule_p=0. # initialise in case we don't find it
                for (rhs,p) in rules:
                    if rhs == [left,right]: # if it's the right rule
                        rule_p=p # get this prob
                #print("rule p: %.5f"%rule_p)
                beta = chart[i]['phrase'][c]['beta']# beta of mother
                #print("beta: %.5f"%beta)
                #for each daughter, calculate the outside prob = p(rule) x beta(mother) x p(sister)
                chart[i]['lex'][left]['beta']+=rule_p*beta*right_p # put the outside prob in the chart
                chart[i+1]['phrase'][right]['beta']+=rule_p*beta*left_p # put the outside prob in the chart
                

    for i in range(len(chart)): # do it for the whole chart
        for c in chart[i]['phrase']:
            beta_cell(i,c)

    #print(chart2string_beta(chart))
    return chart




        

def c_phi(rule,charts,psg,bigram_strings=None,bigram_chain=None,start='S'):
    """
    Calculates the expected counts of a rule in a sentence based on its parses

    Normally we get everything from the same chart, but our parses are split between multiple charts. instead of normalising the whole thing by the probability of the sentence, we normalise each sum of rule uses in each parse by the probability of that parse. The probability of the sentence is the sum of its parses, so we're basically undoing the factoring out of the sentence probabilities in the calculation.

    We also have a special case for MG->mg. This rule has its own prob (1) but it also has the probability of the bigram it represents. As such we also un-factor-out the probability of the rule for each element of the sum. For each use of the rule, we multiply in the probability of its corresponding bigram. Probably we can factor this out too, and multiply by the bigram probs all at once, or maybe their mean?

    Expected counts are normally calculated as follows:
    c_phi(rule,s) = p(rule)/p(s) * (sum of all instances of the rule (beta(lhs) * alpha(rhs1) * alpha(rhs2)))

    Instead we normalise by p(parse) for the sub-sums of rule uses within a parse, and then multiply the whole thing by p(rule). In the special case of MG->mg we also include the rule prob in each individual use prob since it changes by use due to bigram probabilities.

    Arguments
    rule     : (lhs,rhs) pair
    charts   : all filled CKY charts including outside probs for the sentence
    psg      : operations PSG
    bigram_strings : list of bigrams used in each parse, in the same order as the charts
    bigram_chain   : Markhov chain of bigrams
    start          : start category, usually 'S'

    Returns
    expected count of rule given sentence (float): the number of times this rule would be expected to be used in the derivation of this sentence, given the current grammar probs. These can be fractional if the sentence is ambiguous.
    """
    
    p_rule=get_p_rule(rule,psg)
    #print("prob rule: %f"%p_rule)

    (lhs,rhs)=rule
    if rule==('MG',['mg']) and bigram_strings != None and bigram_chain !=None: #special case: need bigram probs
        tot_uses=0.
        for (chart,bis) in zip(charts,bigram_strings): # go through all parses
            #print("bis", bis)
            uses=0.
            k=0 #counter for bigram traversal
            for i in chart: # all rows
                p_parse=get_p_parse(chart,start)

                if lhs in chart[i]['lex']: # lexical cells are always lexical rules
                    mother=chart[i]['lex'][lhs] # 
                    beta=mother['beta'] # outside prob: prob this was used in the parse
                    if beta>0:
                        bps=mother['bp']
                        for r in bps: # go through all backpointers
                            if r==rhs: # this is a bp for our rule
                                if len(r)==1:
                                    p_bi=bigrams[bis[k]][bis[k+1]]
                                    uses+=beta*p_bi # (x p(rule)=1)
                                    k+=1 #increment bigram index
                                    #print ("rhs: %s p: %f"%(rhs,beta))
                                else: print("weird rule")
            tot_uses+=uses*(1/p_parse)
        return tot_uses # we already multiplied by p(rule) by multiplying by bigram probs*1
        
    else: 
        tot_uses=0.
        for chart in charts: # go through all parses
            uses=0.
            for i in chart: # all rows
                p_parse=get_p_parse(chart,start)
                if lhs in chart[i]['phrase']: 
                    mother=chart[i]['phrase'][lhs] # 
                    beta=mother['beta'] # outside prob: prob this was used in the parse
                    if beta>0: # only bother continuing if we'renot going to get 0
                        bps = mother['bp']
                        for r in bps: # go through all backpointers
                            if r==rhs: # this is a bp for our rule
                                if len(r)==2: # branching node: go get inside probs of daughters
                                    left,right=rhs[0],rhs[1]
                                    p_left=chart[i]['lex'][left]['p']
                                    p_right=chart[i+1]['phrase'][right]['p']
                                    use_p = beta*p_left*p_right # calculate prob
                                    #print("rhs: %s %s p: %fx%fx%f=%f"%(left,right,beta,p_left,p_right,use_p))
                                elif len(r)==1: # lexical rule: just use beta
                                    use_p=beta
                                    #print ("lex", rhs)
                                else: print ("error in c_phi: wrong length of backpointer")
                                #print ("p this use: %f"%use_p)
                                uses+=use_p # add in this prob


                if lhs in chart[i]['lex']: # lexical cells are always lexical rules
                    mother=chart[i]['lex'][lhs] # 
                    beta=mother['beta'] # outside prob: prob this was used in the parse
                    if beta>0:
                        bps=mother['bp']
                        for r in bps: # go through all backpointers
                            if r==rhs: # this is a bp for our rule
                                if len(r)==1:
                                    uses+=beta
                                    #print ("rhs: %s p: %f"%(rhs,beta))
                                else: print("weird rule")
            tot_uses+=uses*(1/p_parse)

        #print (tot_uses)
        return p_rule*tot_uses

    

def get_c_phi(rule,parses,psg,bigrams=None,start='S'):
    """wrapper

    Arguments
    rule   : (lhs,rhs)
    parses : list of (operation string, bigram string) pairs
    psg    : operations PSG
    bigrams: bigram markhov chain
    start  : start category

    """
    # get the charts and bigrams from the parses
    charts=[]
    bigram_strings=[]
    for (ops,bis) in parses:
        charts.append(parse(ops,psg,bis,bigrams)) # initial parse with inside probs
        bigram_strings.append(bis)
        
    p_sent = get_p_sent(charts) # probability of the sentence

    for chart in charts:
        p_chart= get_p_parse(chart)
        chart=outside(chart,psg,p_chart/p_sent,start) # outside probs. beta initialised to the proportion of the sentence probability alloted to this parse

    return c_phi(rule,charts,psg,bigram_strings,bigrams,start)


