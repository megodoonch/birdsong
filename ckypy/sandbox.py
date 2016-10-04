

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


grammar_ambig = [("S",[(["S","S"],False),(["a"],False)])]
grammar_ambig_probs = [("S",[(["S","S"],False,0.5),(["a"],False,0.5)])]

grammar_ambig_copy = [("S",[["S","S"],["a"],["S","copy"],["copy","S"]]),("copy",[["copy"]])]
grammar_ambig_copy_probs = [("S",[(["S","S"],0.4),(["a"],0.3),(["S","copy"],0.3)])]


def insert_copies(chart,s):
    n=len(s)
    for j in range(2,n+1):
        print "j=",j
        for i in range(j-1,j/2-1,-1):
            print i,j
            k=2*i-j #start of potential copied material
            print "k=",k
            if k>=0 and s[k:i]==s[i:j]:
                print "copy"
                if "copy" not in chart[i][j]:
                    chart[i][j].append("copy")
    return chart
                        

chart      = [ [ [] for _ in range(n+1) ] for _ in range(n) ]



def parse(sentence,grammar):

   # t0 = time.time()
    # Initialise the chart

    n = len(sentence)
    r = len(grammar)

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
                rhs=rhs[0]
                if rhs==sentence[j-1]:
                    if lhs not in chart[j-1][j]:
                        chart[j-1][j].append(lhs)
                        backpoints[j-1][j][i].append((0,rhs))

        #check if the word is a copy of the previous word
        if j>1 and sentence[j-2]==sentence[j-1]:
            chart[j-1][j].append("copy")
            backpoints[j-1][j][-1].append((0,"copy"))

        

        # Loop over rows, backwards (bottom-up)
        for i in range(j-2,-1,-1): # we start at j-2 because we already did the diagonal

            #deal with copies. We don't need the partitions for this.
            if i >= j/2:
                print i,j
                start=2*i-j #start of potential copied material
                print "start=",start
                if start>=0 and s[start:i]==s[i:j]:
                    print "copy"
                    chart[i][j].append("copy")
                    backpoints[i][j][-1].append((0,"copy"))

                    

            for k in range(i+1,j): # loop over contents of the cell -- partitions, I guess?

                for m,(lhs,rhss) in enumerate(grammar):
                    for rhs in rhss:
                        if len(rhs)==2: # only do non-terminals
                            (rhsB,rhsC)=rhs

                            # Check whether we have the constituents previously recognised
                            if rhsB in chart[i][k] and rhsC in chart[k][j]:
                                if lhs not in chart[i][j]:
                                    chart[i][j].append( lhs )
                                backpoints[i][j][m].append( (k,rhsB,rhsC) )


    #t1 = time.time()
    #print t1-t0
    return (chart,backpoints)










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

