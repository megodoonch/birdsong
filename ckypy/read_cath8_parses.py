


# Ok, here we read parses from the CATH8 corpus and we try to reconstruct them.

# Let's give it a try.


import ckypy
import pandas as pd
import numpy as np



# First, let's read the grammar


dat = pd.DataFrame.from_csv('input/00001_fitted_grammars.txt',sep=' ').reset_index()
rule_probabilities = dat[ dat["grammar"]=="NoCopy" ]


grammar = {}
ruleprobs = {}
for i,row in rule_probabilities.iterrows():
    lhs=row["lhs"]
    rhs=row["rhs"].split(".")
    p  =row["log.prob"]

    #if len(rhs)==1: # if this is a lexical rule
    #    # So as to enter into the format
    #    rhs = [rhs]

    if lhs not in grammar.keys():
        grammar[lhs]=[]
    
    grammar[lhs].append((rhs,False))
    ruleprobs["%s->%s"%(lhs,".".join(rhs))]=p

# Convert back to list so as to ensure that we keep the order of the rules from now on
grammar = grammar.items()

#print grammar









# All right, let's take a sentence and parse it

sentences = pd.DataFrame.from_csv('input/00001_test.txt',sep=' ').reset_index()


sentences["check.prob.nocopy"]=0


for i,row in sentences.iterrows():
    
    sentence = row["sentence"].split(".")
    print sentence

    chart,backpoints = ckypy.parse(sentence,grammar)

    # ckypy.print_chart(chart)

    parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)





    # Suppose we have collected a number of trees, let's draw them!

    # ckypy.tree_to_pdf(tree,"tree.pdf")

    #rules_used = ckypy.rules_used(parses[0],grammar)

    # Ok, let's extract rules used in a particular parse

    sentence_prob = []

    print "# of parses = %i"%len(parses)
    for parse in parses:

        rules_used = ckypy.rules_used(parse)

        log_ps = map(lambda x: ruleprobs[x],rules_used)

        parseprob = sum(log_ps) # the probability of the parse is simply the sum of the log probabilities of the rules used

        sentence_prob.append( parseprob )


    sentences.loc[i,"check.prob.nocopy"]=sentence_prob[0]



corpus_likelihood         = sum(sentences["check.prob.nocopy"])
corpus_likelihood_meaghan = sum(sentences["logprob.NoCopy"])





















# Ok, now for something more challenging: the copy grammar
"""

dat = pd.DataFrame.from_csv('input/00001_fitted_grammars.txt',sep=' ').reset_index()
rule_probabilities = dat[ dat["grammar"]=="Copy+SS" ]


grammar = {}
ruleprobs = {}
for i,row in rule_probabilities.iterrows():
    lhs=row["lhs"]
    rhs=row["rhs"].split(".")
    p  =row["log.prob"]

    #if len(rhs)==1: # if this is a lexical rule
    #    # So as to enter into the format
    #    rhs = [rhs]

    copy = False

    if lhs not in grammar.keys():
        grammar[lhs]=[]

    ruleprobs["%s->%s"%(lhs,".".join(rhs))]=p
    
    if rhs[-1]=="copy": # if this is "the" copy rule
        copy = True
        rhs = rhs[:-1]

    grammar[lhs].append((rhs,copy))

# Convert back to list so as to ensure that we keep the order of the rules from now on
grammar = grammar.items()

#print grammar




sentences = pd.DataFrame.from_csv('input/00001_test.txt',sep=' ').reset_index()
sentences["length"]=np.array([ len(x.split(".")) for x in sentences["sentence"] ])

"""


"""
sentences["check.prob.copy+ss"]=0


if True:
    #sentence = sentences.ix[0]

    sentence = sentences.ix[0]["sentence"].split(".")
    print sentence

    chart,backpoints = ckypy.parse(sentence,grammar)

    # ckypy.print_chart(chart)

    parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)





    # Suppose we have collected a number of trees, let's draw them!

    # ckypy.tree_to_pdf(tree,"tree.pdf")

    #rules_used = ckypy.rules_used(parses[0],grammar)

    # Ok, let's extract rules used in a particular parse

    sentence_prob = []

    print "# of parses = %i"%len(parses)
    for i,parse in enumerate(parses):

        rules_used = ckypy.rules_used(parse)

        ckypy.tree_to_pdf(parse,'output/parse%i.pdf'%i)
        
        log_ps = map(lambda x: ruleprobs[x],rules_used)

        parseprob = sum(log_ps) # the probability of the parse is simply the sum of the log probabilities of the rules used

        sentence_prob.append( parseprob )


    sentences.loc[i,"check.prob.nocopy"]=sentence_prob[0]
"""






"""
sentence = "aiw.aix.aiw.aix".split(".")
"""

#sentence = ["aiw","aiw"]




# sentences[ sentences["length"]<4 ]

# getprob("bbb")





"""
sentences["check.prob.copy+ss"]=0
sentences["n.parses.copy+ss"]=0


for i,row in sentences.iterrows():

    
    sentence = row["sentence"]
    print sentence

    if row["length"]<12:
    #if True:

        p,prob_per_parse = getprob(sentence)

        sentences.loc[i,"check.prob.copy+ss"]=p
        sentences.loc[i,"n.parses.copy+ss"]=len(prob_per_parse)



sentences.to_csv('checkprobs.csv')
"""



# getprob("agb.aaw",output_trees=True)







def getprob(sent,output_trees=False):

    sentence = sent.split(".")

    chart,backpoints = ckypy.parse(sentence,grammar)

    # ckypy.print_chart(chart)

    parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)
    parse_probs = []

    for i,parse in enumerate(parses):
        
        if output_trees:
            ckypy.tree_to_pdf(parse,'output/parse%05i.pdf'%i)

        rules_used = ckypy.rules_used(parse)

        log_ps = map(lambda x: ruleprobs[x],rules_used)

        parseprob = sum(log_ps) # the probability of the parse is simply the sum of the log probabilities of the rules used

        parse_probs.append( parseprob )


    # Add the probabilities of the parses, using a smart
    # bit of algebra to prevent underflow 
    # (essentially what we are trying to compute is log(exp(X)+exp(Y))).
    total_prob = parse_probs[0]
    for logp in parse_probs[1:]:
        total_prob = ckypy.log_add(total_prob,logp)


    return (total_prob,parse_probs)






sentences["check.prob.copy+ss"]=0
sentences["n.parses.copy+ss"]=0


for i,row in sentences.iterrows():

    sentence = row["sentence"].split(".")

    if True:
        print sentence,

        chart,backpoints = ckypy.parse(sentence,grammar)

        nparses_cache = ckypy.n_parses("S",chart,backpoints,grammar,sentence)
        print "Parses (caching): ",nparses_cache,

        probchart,logprob = ckypy.probability("S",chart,backpoints,grammar,sentence,ruleprobs)
        print "Log Probability (with caching): ",logprob

        sentences.loc[i,"check.prob.copy+ss"]=logprob
        sentences.loc[i,"n.parses.copy+ss"]=nparses_cache


        if len(sentence)<7:
        #if True:

            parses = ckypy.collect_trees(0,len(sentence),"S",chart,backpoints,grammar,sentence)
            logprob_fromparses = ckypy.find_prob_from_parses(parses,ruleprobs,output_trees=False)
            print "Log Probability (from parses): ",logprob_fromparses

            nparses = ckypy.n_parses_nocache("S",chart,backpoints,grammar,sentence)
            print "Parses (classic): ",nparses,

            if len(sentence)<10:
                print "N of trees: ",len(parses),

        print

    # ckypy.print_chart(chart)

sentences.to_csv('interim/cath8_sentences.csv')
