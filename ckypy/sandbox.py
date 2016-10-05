

import numpy as np

# Let's try to build a simple CKY parser

#grammar lhs rhs log.prob
#Copy+SS S S.S -1.50220949529
#Copy+SS S S.S.copy -2.0959447686
#Copy+SS S S.T -1.29313372461

#sentence = "aiu.ago.ait"

""" # deprecated form of grammar
grammar = [("S",["NP","VP"]),
           ("VP",["VP","PP"]),
           ("VP",["V","NP"]),
           ("VP",["eats"]),
           ("PP",["P","NP"]),
           ("NP",["Det","N"]),
           ("NP",["she"]),
           ("V" ,["eats"]),
           ("P" ,["with"]),
           ("N" ,["fish"]),
           ("N" ,["fork"]),
           ("Det",["a"])]
nonterminals = set([ a for (a,_) in grammar ])



# Print the chart
def print_chart(ch):
    (n_a,n_b,n_c) = ch.shape
    print "## Chart ##"
    assert n_a==n_b # they should be the same dimension
    for i in range(n_a):
        for j in range(n_b): 
            for k in range(n_c):
                if ch[i][j][k]:
                    lhs,_=grammar[k]
                    print "(%i,%i): %s"%(i,j,lhs)
    print "## end Chart ##"

"""

import ckypy


"""
grammar = [("S",  [["NP","VP"]]),
           ("VP", [["VP","PP"],
                   ["V","NP"],
                   ["eats"]]),
           ("PP", [["P","NP"]]),
           ("NP", [["Det","N"],
                   ["she"]]),
           ("V" , [["eats"]]),
           ("P" , [["with"]]),
           ("N" , [["fish"],
                   ["fork"]]),
           ("Det",[["a"]])]
"""




grammar = [("S",  [["NP","VP"]]),
           ("VP", [["VP","PP"],
                   ["V","NP"],
                   ["eats"]]),
           ("PP", [["P","NP"]]),
           ("NP", [["NP","PP"],
                   ["Det","N"],
                   ["she"]]),
           ("V" , [["eats"]]),
           ("P" , [["with"]]),
           ("N" , [["fish"],
                   ["fork"]]),
           ("Det",[["a"]])]


grammar_copy = [("S",  [(["NP","VP"],False)]),
           ("VP", [(["VP","PP"],False),
                   (["V","NP"],False),
                   (["eats"],False)]),
           ("PP", [(["P","NP"],False)]),
           ("NP", [(["NP","PP"],False),
                   (["Det","N"],False),
                   (["she"],False)]),
           ("V" , [(["eats"],False)]),
           ("P" , [(["with"],False)]),
           ("N" , [(["fish"],False),
                   (["fork"],False)]),
           ("Det",[(["a"],False)])]


grammar_copy_probs = [("S",  [(["NP","VP"],False,1.)]),
           ("VP", [(["VP","PP"],False,0.25),
                   (["V","NP"],False,0.5),
                   (["eats"],False,0.25)]),
           ("PP", [(["P","NP"],False,1.)]),
           ("NP", [(["NP","PP"],False,0.3),
                   (["Det","N"],False,0.4),
                   (["she"],False,0.3)]),
           ("V" , [(["eats"],False,1.)]),
           ("P" , [(["with"],False,1.)]),
           ("N" , [(["fish"],False,0.4),
                   (["fork"],False,0.6)]),
           ("Det",[(["a"],False,1.)])]


grammar_ambig = [("S",[(["S","S"]),(["a"])])]
grammar_ambig_probs = [("S",[(["S","S"],0.5),(["a"],0.5)])]

grammar_ambig_copy = [("S",[["S","S"],["a"],["S","Copy"],["Copy","S"]]),("Copy",[["copy"]])]
grammar_ambig_copy_probs = [("S",[(["S","S"],0.2),(["a"],0.3),(["S","Copy"],0.2),(["Copy","S"],0.3)]),("Copy",[(["copy"],1.)])]





def rule2string(lhs,rhs):
    rule_string=""
    if len(rhs)==1:
        rule_string += "%s->%s"%(lhs,rhs[0])
    elif len(rhs)==2:
        rule_string += "%s->%s.%s"%(lhs,rhs[0],rhs[1])
    return rule_string

def make_rule_probs(g):
    """Given a grammar with rhss (rhs,isCopy,prob) makes dictionary of log rule probs. 
    Keys are strings built from rule names.
    We use the same method for making keys as is used in the parser in case we want to change it"""
    rule_probs={}
    for (lhs,rhss) in g:
        for (rhs,p) in rhss:
            rule_probs[rule2string(lhs,rhs)]=np.log(p)
    return rule_probs



copy_lhs = "Copy"
copy_rhs = "copy"



def parse(sentence,grammar):

   # t0 = time.time()
    # Initialise the chart

    n = len(sentence)
    r = len(grammar)

    #check the status of copies in this grammar
    copy_rule = (copy_lhs,[[copy_rhs]])
    copy_grammar = copy_rule in grammar
    if copy_grammar:
        copy_n = grammar.index(copy_rule)


    # The CKY Chart
    chart      = [ [ [] for _ in range(n+1) ] for _ in range(n) ]

    # The back pointers: tells us how we made things.
    # in particular, backpoints[i][j][m] tells us all the ways
    # in which we made an item of "category" m (the m-th left-hand side in the grammar)
    # each element (k,catB,catC) tells us that made the item using the m-th rule,
    # from categories B and C (that should identify uniquely the rule used)
    # where B was from i to k and C was from k to j.

    backpoints = [ [ [ [] for _ in range(r) ] for _ in range(n+1) ] for _ in range(n) ]

    # Lots of inspiration for this came from courses.washington.edu/ling571/ling571_fall_2010/slides/cky_cnf.pdf
    #this is a different implementation from Meaghan's earlier version that started by initialising the diagonal and then looped over rows.
    for j in range(1,n+1): ## loop over columns, fill them from the bottom up
        # Put the initial values for the diagonal cell in this column
        for i,(lhs,rhss) in enumerate(grammar):
            for rhs in rhss:
                if rhs==[sentence[j-1]]:
                    if lhs not in chart[j-1][j]:
                        chart[j-1][j].append(lhs)
                        backpoints[j-1][j][i].append((0,rhs))

        #check if the word is a copy of the previous word
        if copy_grammar:
            if j>1 and sentence[j-2]==sentence[j-1]:
                chart[j-1][j].append(copy_lhs)
                backpoints[j-1][j][copy_n].append((0,[copy_rhs])) 
        

        # Loop over rows, backwards (bottom-up)
        for i in range(j-2,-1,-1): # we start at j-2 because we already did the diagonal

            #deal with copies. We don't need the partitions for this.
            if copy_grammar:
                if i >= j/2:
                    # print i,j
                    start=2*i-j #start of potential copied material
                    # print "start=",start
                    if start>=0 and s[start:i]==s[i:j]:
                        # print "copy"
                        chart[i][j].append(copy_lhs)
                        backpoints[i][j][copy_n].append((0,[copy_rhs]))

                    

            for k in range(i+1,j): # loop over contents of the cell -- partitions, I guess?

                for m,(lhs,rhss) in enumerate(grammar):
                    for rhs in rhss:
                        if len(rhs)==2: # only do non-terminals
                            (rhsB,rhsC)=rhs

                            # Check whether we have the constituents previously recognised
                            if rhsB in chart[i][k] and rhsC in chart[k][j]:
                                if lhs not in chart[i][j]:
                                    chart[i][j].append( lhs )
                                backpoints[i][j][m].append( (k,[rhsB,rhsC]) )
    #t1 = time.time()
    #print t1-t0
    return (chart,backpoints)


def collect_trees(from_i,to_i,category,chart,backpoints,grammar,sentence):

    # Let's go collect trees!

    def reconstruct(i,j,category):
        # GIVE ALL RECONSTRUCTIONS OF THE category between i and j, using back pointers 
        # (this will be called recursively.
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

        category_n = [ lhs for (lhs,_) in grammar ].index(category)

        reconstructions = []

        # For each of the reconstructions, collect the trees
        for (k,rhs) in backpoints[i][j][category_n]:

            assert (len(rhs)>0 and len(rhs)<3)

            if len(rhs)==1:
                lhs,rhss=grammar[category_n]
                if category==lhs and rhs in rhss:
                    reconstructions.append( (rule2string(lhs,rhs),[]) )

            else:
                (rhsB,rhsC)=(rhs[0],rhs[1])
                # First let's reconstruct each of the subtrees
                trees_rhsB = reconstruct(i,k,rhsB)
                trees_rhsC = reconstruct(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                rulestr = rule2string(category,[rhsB,rhsC])#"%s->%s.%s"%(category,rhsB,rhsC)
                #if isCopy: rulestr+=".copy"
                for reconstrB in trees_rhsB:
                    for reconstrC in trees_rhsC:

                        # And then put them together
                        reconstruction = (rulestr,
                                          [reconstrB,
                                           reconstrC])
                        reconstructions.append( reconstruction )
                    
        return reconstructions
                        
    return reconstruct(from_i,to_i,category)
                        


def n_parses(category,chart,backpoints,grammar,sentence,from_i=0,to_i=None):
    # Find the number of parses for the sentence and the given category

    # Let's go collect trees!
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
        #if not( category in chart[i][j]):
        #    print category
        #    print "not in"
        #    print "chart[%i][%i]"%(i,j)
        #    print chart[i][j]

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
                #return 1

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
    # Find the probability of the parses for the sentence and the given category

    # Let's go collect trees!
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
                log_prob = ckypy.log_add(log_prob,prob_rhsB+prob_rhsC+ruleprob)
                # mind you, the probability of this item is the PRODUCT of the two children,
                # hence the addition here, and of course the probability of applying the rule.
                log_probs[i][j][category_n] = log_prob
         
 
                    
        return log_prob
    
    return log_probs,get_prob(from_i,to_i,category)
    

















probs = ckypy.make_rule_probs(grammar_copy_probs)
ambig_probs = ckypy.make_rule_probs(grammar_ambig_probs)



sentence = "she eats a fish with a fork".split(" ")

#let's make some catalan numbers for testing

def a_sent(n): return ["a"]*n

#catalan numbers are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420, 24466267020, 91482563640, 343059613650, 1289904147324, 4861946401452, 
def catalan(n):
    """calculates the first n catalan numbers (except 0th) extremely inefficiently"""
    for i in range(1,n):
        s=a_sent(i)
        chart,backpoints = ckypy.parse(s,grammar_ambig)
        print(ckypy.n_parses("S",chart,backpoints,grammar_ambig,s))


def catalan_probs(n):
    """calculates the probabilities of fully ambiguous trees, with both rules p=0.5"""
    for i in range(1,n):
        s=a_sent(i)
        chart,backpoints = ckypy.parse(s,grammar_ambig)
        print(ckypy.probability("S",chart,backpoints,grammar_ambig,s,ambig_probs)[1])
    



#a_sentence = "a a a a a a a a a a a a a a a a a".split(" ")

#list_times=[]
#for i in range(1000):
#    list_times.append(ckypy.parse(sentence,grammar_ambig))
#    i+=1

#set_times=[]
#for i in range(1000):
#    set_times.append(ckypy.parse_set(sentence,grammar_ambig) )
#    i+=1
#
#print (sum(list_times)/float(len(list_times)))
#print (sum(set_times)/float(len(set_times)))




chart,backpoints = ckypy.parse(sentence,grammar_copy)

# ckypy.print_chart(chart)

#don't do this for grammar_ambig with a long sentence. It takes forever.
parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar_copy,sentence)


for i,parse in enumerate(parses):
    
    ckypy.tree_to_pdf(parse,"parse_%i.pdf"%i)


ckypy.probability("S",chart,backpoints,grammar_ambig,a_sentence,ambig_probs)


# Now let's do tree collection

# We should have found that chart(0,len(sentence)) contains S.
# How could it have been built?


trees = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,sentence)




reload(ckypy)
for x in range(1,20):
    print ckypy.log_add(0.5, 2*(10**(-x) ))

