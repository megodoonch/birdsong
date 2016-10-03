

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


def make_rule_probs(g):
    """Given a grammar with rhss (rhs,isCopy,prob) makes dictionary of log rule probs. 
    Keys are strings built from rule names."""
    rule_probs={}
    for (lhs,rhss) in g:
        for (rhs,isCopy,p) in rhss:
            if len(rhs)==1:
                rule_string = "%s->%s"%(lhs,rhs[0])
            elif len(rhs)==2:
                rule_string = "%s->%s.%s"%(lhs,rhs[0],rhs[1])
            if isCopy:
                rule_string = rule_string.append(".copy")
            rule_probs[rule_string]=np.log(p)
    return rule_probs


probs = make_rule_probs(grammar_copy_probs)



sentence = "she eats a fish with a fork".split(" ")




chart,backpoints = ckypy.parse(sentence,grammar_copy)

# ckypy.print_chart(chart)

parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)


for i,parse in enumerate(parses):
    
    ckypy.tree_to_pdf(parse,"parse_%i.pdf"%i)


ckypy.probability("S",chart,backpoints,grammar_copy,sentence,probs)




# Now let's do tree collection

# We should have found that chart(0,len(sentence)) contains S.
# How could it have been built?


trees = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,sentence)




reload(ckypy)
for x in range(1,20):
    print ckypy.log_add(0.5, 2*(10**(-x) ))

