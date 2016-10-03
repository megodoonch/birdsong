

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



def make_rule_probs(g):
    """Given a grammar with rhss (rhs,isCopy,prob) makes dictionary of log rule probs. 
    Keys are strings built from rule names.
    We use the same method for making keys as is used in the parser in case we want to change it"""
    rule_probs={}
    for (lhs,rhss) in g:
        for (rhs,isCopy,p) in rhss:
            rule_probs[ckypy.rule2string(lhs,rhs,isCopy)]=np.log(p)
    return rule_probs


probs = make_rule_probs(grammar_copy_probs)
#probs = make_rule_probs(grammar_ambig_probs)



sentence = "she eats a fish with a fork".split(" ")

#sentence = "a a a a a a a a a a a a a a a a a".split(" ")

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




chart,backpoints = ckypy.parse_set(sentence,grammar_ambig)

# ckypy.print_chart(chart)

parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar_ambig,sentence)


for i,parse in enumerate(parses):
    
    ckypy.tree_to_pdf(parse,"parse_%i.pdf"%i)


ckypy.probability("S",chart,backpoints,grammar_copy,sentence,probs)


log_probs = np.zeros( (5, 5+1, 10), dtype="Float64" )


# Now let's do tree collection

# We should have found that chart(0,len(sentence)) contains S.
# How could it have been built?


trees = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,sentence)




reload(ckypy)
for x in range(1,20):
    print ckypy.log_add(0.5, 2*(10**(-x) ))

