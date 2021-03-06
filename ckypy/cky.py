"""
This CKY parser handles copies as follows:

Copying is a string operation. We do this with a 2-stage grammar. The first grammar generates strings that include the word "copy". The second grammar generates strings with copies of arbitrary amounts of material from before the word "copy" in place of the word "copy".

sources:
http://windowoffice.tumblr.com/post/33548509/logsum-underflow-trick-re-discovery
https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
https://gist.github.com/longouyang/3504979
http://mikelove.wordpress.com/2011/06/06/log-probabilities-trick/
"""

import numpy as np
from common import *
from pandas import DataFrame


############### UTILITY FUNCTIONS ######################

# read-only global variables. Names for these special copy category and "lexical item"s for copies
COPY_LHS = "Copy"
COPY_RHS = "copy"


 
def grammar2string_probs(grammar):
    """
    Prints a probabilistic grammar as a readable string.
    
    Arguments:
    grammar : a grammar with probabilities
    """
    s = ""
    for (lhs,rhss) in grammar:
        for (rhs,p) in rhss:
            s+=rule2string(lhs,rhs)+" %.5f\n"%p
    return s


def backpointer_cell_to_string(cell):
    """Makes a string of a cell of a backpointer chart"""
    s="["
    for (k,rhs) in cell:
        s+="(%i, %s)"%(k,",".join(rhs))
    s+="]"
    return s



def make_backpointers_pretty(ch,g):
    """Turn backpointers into a nice list of lists for DataFrame to read

    Arguments:
    ch : backpointers chart
    g  : grammar
    """
    lhss = [lhs for lhs,_ in g]
    g_length = len(g)
    s=[["LHS:"]+lhss]
    for i,row in enumerate(ch):
        for j,col in enumerate(ch[i]):
            if any(len(ch[i][j][m])>0 for m in range(g_length)) :
                e=["(%i,%i)"%(i,j)]
                for cell in ch[i][j]:
                    e+=[backpointer_cell_to_string(cell)]
                s+=[e]
    return s


def pretty_print_backpointers(ch,g):
    """
    uses DataFrame from Pandas to print a nicely laid out backpointer chart

    Arguments:
    ch : backpointer chart
    g  : grammar
    """
    ch=make_backpointers_pretty(ch,g)
    print ("#### BACKPOINTERS ####")
    print (DataFrame(ch))
    print ("#### end BACKPOINTERS ####")








############# MAIN FUNCTIONS #################

def parse(sentence,grammar):

    """
    This is the parser. It behaves like a normal CKY parser except that whereever possible we add an instance of the rule Copy->copy. This is possible just in case the substring for that cell is an exact copy of the substring immediately before it. Once these Copy categories are in place, they are treated like any other catagory.

    We will follow the style used in courses.washington.edu/ling571/ling571_fall_2010/slides/cky_cnf.pdf in that instead of initialising the chart with the lexical items on the diagonal and then proceeding row by row, instead we proceed column by column from the bottom up, initialising the bottom cell of the column (ie the one on the diagonal) as we go. We also initialise the copies as we go.

    Arguments:
    sentence : string list
    grammar  : (category * catagory list) list  -- *note this is not a grammar with probabilities* this grammar has a set of right hand sides for each left hand side.

    output   : (chart, backpointers)
    chart        : the filled CKY chart, a list of lists of lists of categories. 
    backpointers : list of list of lists of (partition, rhs) pairs. each cell of the CKY chart has a corresponding cell in the backpointers. In a backpointer cell, each LHS in the grammar has its own list. Into that list we put all uses of that rule set. An entry is a (partition, rhs) pair.
    """

    # Initialise the chart

    n = len(sentence)
    r = len(grammar)

    #check the status of copies in this grammar
    copy_rule = (COPY_LHS,[[COPY_RHS]])
    copy_grammar = copy_rule in grammar
    if copy_grammar:
        copy_n = grammar.index(copy_rule) # this is the index of the copy rule in the grammar



    # The CKY Chart
    chart      = [ [ [] for _ in range(n+1) ] for _ in range(n) ]

    # The back pointers: tells us how we made things.
    # in particular, backpoints[i][j][m] tells us all the ways
    # in which we made an item of "category" m (the m-th left-hand side in the grammar)
    # each element (k,rhs) tells us that made the item using the m-th rule set,
    # from the right hand side rhs. (that should identify uniquely the rule used)
    # if the rhs is binary -- say [B,C] then
    # B was from i to k and C was from k to j.
    # if the rhs was lexical, we let k=0

    backpoints = [ [ [ [] for _ in range(r) ] for _ in range(n+1) ] for _ in range(n) ]

    for j in range(1,n+1): ## loop over columns, fill them from the bottom up
        # Put the initial values for the diagonal cell in this column
        for i,(lhs,rhss) in enumerate(grammar):
            for rhs in rhss:
                if rhs==[sentence[j-1]]:
                    if lhs not in chart[j-1][j]:
                        chart[j-1][j].append(lhs)
                        backpoints[j-1][j][i].append((0,rhs)) # k=0 since there's no partition

        #check if the word is a copy of the previous word
        if copy_grammar:
            if j>1 and sentence[j-2]==sentence[j-1]:
                chart[j-1][j].append(COPY_LHS)
                backpoints[j-1][j][copy_n].append((0,[COPY_RHS]))  # k=0 since there's no partition
        
        # Loop over rows, backwards (bottom-up)
        for i in range(j-2,-1,-1): # we start at j-2 because we already did the diagonal

            #deal with copies. We don't need the partitions for this.
            if copy_grammar:
                if i >= j/2: # if there's enough room for a copy this length before this cell
                    start=2*i-j #start of potential copied material
                    if start>=0 and sentence[start:i]==sentence[i:j]: 
                        # make sure we're not trying to start before the sentence starts! 
                        # Check if it's a copy.
                        chart[i][j].append(COPY_LHS)
                        backpoints[i][j][copy_n].append((0,[COPY_RHS]))# k=0 since there's no partition
                        
            for k in range(i+1,j): # "loop over contents of the cell" -- partitions, I guess?

                for m,(lhs,rhss) in enumerate(grammar):
                    for rhs in rhss:
                        if len(rhs)==2: # only do non-terminals
                            (rhsB,rhsC)=rhs

                            # Check whether we have the constituents previously recognised
                            if rhsB in chart[i][k] and rhsC in chart[k][j]:
                                if lhs not in chart[i][j]: #only write it once
                                    chart[i][j].append( lhs )
                                backpoints[i][j][m].append( (k,[rhsB,rhsC]) )
    return (chart,backpoints)


def collect_trees(category,chart,backpoints,grammar,sentence,from_i=0,to_i=None):
    """
    Once we've parsed the sentence we can collect the trees.

    Arguments:
    category   : usually this is the start category, but we can check for any category on any interval
    chart      : category list list list -- the CKY chart
    backpoints : (int * category list) list list list -- the backpointers from the parse
    grammar    : (category * catagory list) list  -- the same grammar as from the parse
    sentence   : string list
    from_i     : int -- start of the span you're interested in. Default 0 (start of sent)
    to_i       : int -- end of the span you're interested in. Default len(sent) (end of sent)

    output     : list of trees. Trees are tuples of (rule used, list of daughter trees)
    """
    
    if to_i==None:
        to_i=len(sentence)

    def reconstruct(i,j,category):
        # GIVE ALL RECONSTRUCTIONS OF THE category between i and j, using back pointers 
        # (this will be called recursively.
        # We do this "inner function" kind of construction so that we don't need to keep
        # passing along copies of the backpointers and chart

        #print "Reconstructing (%i,%i) %s"%(i,j,category)

        # We assume that chart[from_i][to_i] contains category!!
        assert category in chart[i][j]
 
        # we need the index of the LHS in the grammar because we'll put the backpointers in the nth list in that cell
        category_n = [ lhs for (lhs,_) in grammar ].index(category)

        reconstructions = []

        # For each of the reconstructions, collect the trees
        for (k,rhs) in backpoints[i][j][category_n]:

            assert (len(rhs)>0 and len(rhs)<3) # make sure our rule's not messed up

            # for lexical items
            if len(rhs)==1:
                lhs,rhss=grammar[category_n]
                if category==lhs and rhs in rhss: # we need to check because the copies might not really work, if the grammar doesn't allow them in this spot.
                    #reconstructions.append( (rule2string(lhs,rhs),[]) )
                    if rhs[0]==COPY_RHS:
                        reconstructions.append( ("%s\n%s\n%s"%(lhs,rhs[0],('.'.join(sentence[i:j]))),[]) )
                    else:
                        reconstructions.append( ("%s\n%s"%(lhs,rhs[0]),[]) )

            else:
                (rhsB,rhsC)=(rhs[0],rhs[1])
                # First let's reconstruct each of the subtrees
                trees_rhsB = reconstruct(i,k,rhsB)
                trees_rhsC = reconstruct(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                rulestr = rule2string(category,[rhsB,rhsC])#"%s->%s.%s"%(category,rhsB,rhsC)
                for reconstrB in trees_rhsB:
                    for reconstrC in trees_rhsC:

                        # And then put them together
                        reconstruction = (category,  #rulestr,
                                          [reconstrB,
                                           reconstrC])
                        reconstructions.append( reconstruction )
                    
        return reconstructions
                        
    return reconstruct(from_i,to_i,category)
                        


def n_parses(category,chart,backpoints,grammar,sentence,from_i=0,to_i=None):
    """ Find the number of parses for the sentence and the given category
    Similar to collect_trees, except it just counts instead of storing the trees

    Arguments:
    category   : usually this is the start category, but we can check for any category on any interval
    chart      : category list list list -- the CKY chart
    backpoints : (int * category list) list list list -- the backpointers from the parse
    grammar    : (category * catagory list) list  -- the same grammar as from the parse
    sentence   : string list
    from_i     : int -- start of the span you're interested in. Default 0 (start of sent)
    to_i       : int -- end of the span you're interested in. Default len(sent) (end of sent)

    output     : int -- number of parses
   """
    if to_i==None:
        to_i=len(sentence)

    # This is a chart where we keep track of how many parses each item has:
    #  element (n,m,r) means that there are X parses for an item between n and m
    #  being of category identified by the LHS of rule r.
    # This will enable us not to have to re-do such calculations later in our recursions.
    n = len(sentence)
    r = len(grammar)
    n_parses = np.zeros( (n, n+1, r), dtype="int" )


    def n_reconstructions(i,j,category):
        # Find how many reconstructions there are for the given category between i and j,
        # using back pointers (this will be called recursively).
        # We do this "inner function" kind of construction so that we don't need to keep
        # passing along copies of the backpointers and chart

        #print "Reconstructing (%i,%i) %s"%(i,j,category)

        # We assume that chart[from_i][to_i] contains category!!
        assert category in chart[i][j]

        category_n = [ lhs for (lhs,_) in grammar ].index(category)

        if n_parses[i][j][category_n]!=0: # if we've already calculated this, just return the cached number
            return n_parses[i][j][category_n]
 

        # For each of the reconstructions, collect the trees
        reconstructions = 0
        for (k,rhs) in backpoints[i][j][category_n]:

            assert (len(rhs)>0 and len(rhs)<3)

            if len(rhs)==1:
                lhs,rhss=grammar[category_n]
                if category==lhs and rhs in rhss:
                    reconstructions +=1
                    n_parses[i][j][category_n] = reconstructions

            else:
                
                (rhsB,rhsC)=(rhs[0],rhs[1])
                # First let's reconstruct each of the subtrees
                trees_rhsB = n_reconstructions(i,k,rhsB)
                trees_rhsC = n_reconstructions(k,j,rhsC)

                # the number of ways you can make the children
                reconstructions+=trees_rhsB*trees_rhsC

            n_parses[i][j][category_n] = reconstructions
        return reconstructions

    return n_reconstructions(from_i,to_i,category)
    


def probability(category,chart,backpoints,grammar,sentence,ruleprobs,from_i=0,to_i=None):
    """ Finds the probability of the parses for the sentence and the given category
    
    Arguments:
    category   : start category, usually "S"
    chart      : filled CKY chart
    backpoints : chart of backpointers
    grammar    : the grammar used to fill the chart. Probs are separate.
    sentence   : string list
    ruleprobs  : dictionary. Keys are strings, one for each rule "lhs->rhs". 
                 Values are log probabilities.
                 Probs for all rules with the same LHS sum to 1 (log-sum to 0)
    from_i     : starting point, usually the start of the sentence but you can also test a substring
    to_i       : end point, usually the end of the sentence but you can also test a substring

    """

    # if there's no endpoint given, default to the end of the sentence.
    if to_i==None:
        to_i=len(sentence)

    # This is a chart where we keep track of the probability of each item: 
    #  element (n,m,r) indicates the probability for an item between n and m
    #  being of category identified by the LHS of rule r.
    # This will enable us not to have to re-do such calculations later in our recursions.
    n = len(sentence)
    r = len(grammar) #number of lhs
    log_probs = np.zeros( (n, n+1, r), dtype="Float64" ) # makes an n x n+1 x r array of 0.'s
    # We store the probability that each category (LHS) spans every interval within the sentence


    def get_prob(i,j,category):
        # Find how many reconstructions there are for the given category between i and j,
        # using back pointers (this will be called recursively).
        # We do this "inner function" kind of construction so that we don't need to keep
        # passing along copies of the backpointers and chart

        #print "Reconstructing (%i,%i) %s"%(i,j,category)

        # We assume that chart[from_i][to_i] contains category!!
        assert category in chart[i][j]
        #if not( category in chart[i][j]):
        #    print category
        #    print "not in"
        #    print "chart[%i][%i]"%(i,j)
        #    print chart[i][j]

        # get the index of the category as lhs in the grammar so we know where to put its probs in the innermost array 
        category_n = [ lhs for (lhs,_) in grammar ].index(category)

        if log_probs[i][j][category_n]!=0: # if we've already calculated this, just return the cached number
            return log_probs[i][j][category_n]

        #initialise total prob
        log_prob=None

        # For each of the reconstructions, collect the trees
        for (k,rhs) in backpoints[i][j][category_n]:

            assert (len(rhs)>0 and len(rhs)<3)

            if len(rhs)==1:
                lhs,rhss=grammar[category_n]
                if category==lhs and rhs in rhss:
                    rule = rule2string(lhs,rhs)
                    log_prob = ruleprobs[rule] #get the prob from the prob grammar
                    log_probs[i][j][category_n] = log_prob #add it to log_probs

            else:
                (rhsB,rhsC)=(rhs[0],rhs[1])
                # First let's reconstruct each of the subtrees
                prob_rhsB = get_prob(i,k,rhsB)
                prob_rhsC = get_prob(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                rulestr = rule2string(category,[rhsB,rhsC])#"%s->%s.%s"%(category,rhsB,rhsC)
                ruleprob = ruleprobs[rulestr]
                log_prob = log_add(log_prob,prob_rhsB+prob_rhsC+ruleprob)
                # mind you, the probability of this item is the PRODUCT of the two children,
                # hence the addition here, and of course the probability of applying the rule.
                log_probs[i][j][category_n] = log_prob
         
 
                    
        return log_prob
    
    return log_probs,get_prob(from_i,to_i,category)
    

