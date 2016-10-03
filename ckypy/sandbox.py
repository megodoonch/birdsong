

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







sentence = "she eats a fish with a fork".split(" ")




chart,backpoints = ckypy.parse(sentence,grammar)

# ckypy.print_chart(chart)

parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)


for i,parse in enumerate(parses):
    
    ckypy.tree_to_pdf(parse,"parse_%i.pdf"%i)






# Now let's do tree collection

# We should have found that chart(0,len(sentence)) contains S.
# How could it have been built?


trees = collect_trees(0,len(sentence),"S",chart,backpoints,sentence)




reload(ckypy)
for x in range(1,20):
    print ckypy.log_add(0.5, 2*(10**(-x) ))

