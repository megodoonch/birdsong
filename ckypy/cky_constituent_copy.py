

# Let's make a simple module that can do CKY parsing
# and tree collection




import numpy as np
 
"""
sources:
http://windowoffice.tumblr.com/post/33548509/logsum-underflow-trick-re-discovery
https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
https://gist.github.com/longouyang/3504979
http://mikelove.wordpress.com/2011/06/06/log-probabilities-trick/
"""
def log_add(logx,logy):
    # This adds two log-transformed variables,
    # taking care of the underflow that you usually find when you do this

    if logx==None: # This is just a hack so that I can give a not-defined sum
        return logy

    # First, make X the maximum
    if (logy > logx):
        logx,logy = logy,logx
        #temp = logx
        #logx = logy
        #logy = temp

    # How far "down" is logY from logX?
    negdiff = logy - logx
    if negdiff < -30: # If it's small, we can just ignore logY altogether (it won't make much of a difference)
        return logx
        # However, in my case I can maybe keep it in because it will just become zero in the sum below.

    # Otherwise, use some simple algebra to stay in the log domain
    # (i.e. here we use log(X)+log(Y) = log(X)+log(1.0+exp(log(Y)-log(X)))
    return logx + np.log(1.0 + np.exp(negdiff))
 

def rule2string(lhs,rhs,isCopy=False):
    rule_string=""
    if len(rhs)==1:
        rule_string += "%s->%s"%(lhs,rhs[0])
    elif len(rhs)==2:
        rule_string += "%s->%s.%s"%(lhs,rhs[0],rhs[1])
    if isCopy:
        rule_string+=".copy"
    return rule_string



def print_chart(ch):
    print "### Chart ###"
    for i,row in enumerate(ch):
        for j,col in enumerate(ch[i]):
            if len(ch[i][j])>0:
                print "(%i,%i)"%(i,j),ch[i][j]
    print "### end Chart ###"

def print_backpointers(ch):
    g_length = len(ch[0][0])
    print "### Backpointers ###"
    for i,row in enumerate(ch):
        for j,col in enumerate(ch[i]):
            if any(len(ch[i][j][m])>0 for m in range(g_length)) :
                print "(%i,%i)"%(i,j),ch[i][j]
    print "### end Backpointers ###"





import time

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
            for rhs,_ in rhss:
                if rhs==[sentence[j-1]]:
                    if lhs not in chart[j-1][j]:
                        chart[j-1][j].append(lhs)


        # Loop over rows, backwards (bottom-up)
        for i in range(j-2,-1,-1):

            for k in range(i+1,j): # loop over contents of the cell -- partitions, I guess?

                for m,(lhs,rhss) in enumerate(grammar):
                    for rhs,isCopy in rhss:
                        if len(rhs)==2: # only do non-terminals
                            (rhsB,rhsC)=rhs

                            # Check whether we have the constituents previously recognised
                            if rhsB in chart[i][k] and rhsC in chart[k][j]:
                                
                                if isCopy: # if this is a copy rule, check that the two items are identical
                                    valid_parse = (sentence[i:k] == sentence[k:j])
                                else:
                                    valid_parse = True # if this is not a copy rule, we can just continue

                                if valid_parse:
                                    if lhs not in chart[i][j]:
                                        chart[i][j].append( lhs )
                                    backpoints[i][j][m].append( (k,rhsB,rhsC,isCopy) )


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

        if len(backpoints[i][j][category_n])==0:
            # If there are no back pointers, this must be a lexical item.
            lhs,_ = grammar[category_n]
            rhs   = ".".join(sentence[i:j])
            return [(rule2string(lhs,[rhs],False),[])]
            #return [("%s->%s"%(lhs,rhs),[])]
            # else something is seriously wrong!
            
        else:
            reconstructions = []

            # For each of the reconstructions, collect the trees
            for (k,rhsB,rhsC,isCopy) in backpoints[i][j][category_n]:

                # First let's reconstruct each of the subtrees
                trees_rhsB = reconstruct(i,k,rhsB)
                trees_rhsC = reconstruct(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                rulestr = rule2string(category,[rhsB,rhsC],isCopy)#"%s->%s.%s"%(category,rhsB,rhsC)
                #if isCopy: rulestr+=".copy"
                for reconstrB in trees_rhsB:

                    if isCopy:
                        # If this is a use of the copy rule, only count the making
                        # of one side (e.g. the left side) and count that copy just
                        # instantly put the other one there, without needing to apply any further rules.
                        reconstruction = (rulestr,
                                          [reconstrB])
                        reconstructions.append( reconstruction )

                    else:
                    
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
            

        if len(backpoints[i][j][category_n])==0:
            # If there are no back pointers, this must be a lexical item.
            n_parses[i][j][category_n] = 1
            return 1
            #lhs,_ = grammar[category_n]
            #rhs   = ".".join(sentence[i:j])
            #return [("%s->%s"%(lhs,rhs),[])]
            # else something is seriously wrong!
            
        else:
            reconstructions = 0

            # For each of the reconstructions, collect the trees
            for (k,rhsB,rhsC,isCopy) in backpoints[i][j][category_n]:

                # First let's reconstruct each of the subtrees
                trees_rhsB = n_reconstructions(i,k,rhsB)
                trees_rhsC = n_reconstructions(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                #rulestr = "%s->%s.%s"%(category,rhsB,rhsC)
                #if isCopy: rulestr+=".copy"
                if isCopy:
                    reconstructions += trees_rhsB
                else:
                    reconstructions += trees_rhsB*trees_rhsC 
                    # the number of ways you can make the children

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
            
        if len(backpoints[i][j][category_n])==0:
            # If there are no back pointers, this must be a lexical item.
            # So it can't be a copy rule at least, whew!

            # Ok, this is admittedly ugly but it will do for now
            # make a string that represents the rule. This will match a key in the ruleprobs dict.
            lhs,_ = grammar[category_n]
            rhs   = ".".join(sentence[i:j])
            rule = rule2string(lhs,[rhs],False) #"%s->%s"%(lhs,rhs)

            # We've got the p; cache it and return it.
            logp =  ruleprobs[rule] #get it
            log_probs[i][j][category_n] = logp #add it to log_probs
            return logp
            
        else: # if it's got backpointers
            log_prob = None

            # For each of the reconstructions, calculate the prob (we have to ADD those because they are "ambiguities")
            for (k,rhsB,rhsC,isCopy) in backpoints[i][j][category_n]:

                # First let's reconstruct each of the subtrees
                # I think this is where the cache comes in. We're about to get all recursive on your ass.
                prob_rhsB = get_prob(i,k,rhsB)
                prob_rhsC = get_prob(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                rulestr = rule2string(category,[rhsB,rhsC],isCopy) #"%s->%s.%s"%(category,rhsB,rhsC) # make the key for look-up in ruleprobs
                #if isCopy: rulestr+=".copy"
                ruleprob = ruleprobs[rulestr]

                if isCopy:
                    log_prob = log_add(log_prob,
                                       # Mind you, this is the probability of making the child TIMES the prob
                                       # of applying this rule (the copy rule). Since we're in log probabilities,
                                       # we add the probs.
                                       prob_rhsB+ruleprob)
                else:
                    # We SUM the different ways of making this item
                    log_prob = log_add(log_prob,
                                       # mind you, the probability of this item is the PRODUCT of the two children,
                                       # hence the addition here, and of course the probability of applying the rule.
                                       prob_rhsB+prob_rhsC+ruleprob)
 
                   
            log_probs[i][j][category_n] = log_prob
            return log_prob
            #we define get_prob not only to update log_probs but also to return the prob of the interval and category (usually the whole sentence and the start category) so that the main function can return that probability

    
    return log_probs,get_prob(from_i,to_i,category)
    #return get_prob(from_i,to_i,category)
    












def n_parses_nocache(category,chart,backpoints,grammar,sentence,from_i=0,to_i=None):
    # This is really just for backwards compatibility checking.
    # Find the number of parses for the sentence and the given category

    # Let's go collect trees!
    if to_i==None:
        to_i=len(sentence)


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


        if len(backpoints[i][j][category_n])==0:
            # If there are no back pointers, this must be a lexical item.
            return 1
            #lhs,_ = grammar[category_n]
            #rhs   = ".".join(sentence[i:j])
            #return [("%s->%s"%(lhs,rhs),[])]
            # else something is seriously wrong!
            
        else:
            reconstructions = 0

            # For each of the reconstructions, collect the trees
            for (k,rhsB,rhsC,isCopy) in backpoints[i][j][category_n]:

                # First let's reconstruct each of the subtrees
                trees_rhsB = n_reconstructions(i,k,rhsB)
                trees_rhsC = n_reconstructions(k,j,rhsC)

                # Ok, then we should combine all reconstructions from both sides of the rule
                #rulestr = "%s->%s.%s"%(category,rhsB,rhsC)
                #if isCopy: rulestr+=".copy"
                if isCopy:
                    reconstructions += trees_rhsB
                else:
                    reconstructions += trees_rhsB*trees_rhsC 
                    # the number of ways you can make the children

            return reconstructions

    return n_reconstructions(from_i,to_i,category)
    










def rules_used(parse):
    # In a particular parse, give a listing of all the rules used.
    # We just use a simple shorthand notation %s->%s.%s to identify
    # each rule.

    (rule,children) = parse

    rules = []
    for child in children:
        rules += rules_used(child)

    rules += [rule]
    return rules






def find_prob_from_parses(parses,ruleprobs,output_trees=False):

    # Find the probability of a particular sentence from its parses.
    # I.e. this will be untractable for big guys but I can use it to
    # verify my faster implementation (using caching) on the smaller guys.

    parse_probs = []

    for i,parse in enumerate(parses):
        
        if output_trees:
            tree_to_pdf(parse,'output/parse%05i.pdf'%i)

        rules = rules_used(parse)
        log_ps = map(lambda x: ruleprobs[x],rules)
        parseprob = sum(log_ps) # the probability of the parse is simply the sum of the log probabilities of the rules used
        parse_probs.append( parseprob )

    # Add the probabilities of the parses, using a smart
    # bit of algebra to prevent underflow 
    # (essentially what we are trying to compute is log(exp(X)+exp(Y))).
    total_prob = parse_probs[0]
    for logp in parse_probs[1:]:
        total_prob = log_add(total_prob,logp)

    return total_prob






def get_nodes_edges(tree,prefix=""):
    # Given a particular tree, define a list of nodes and edges
    # so that we can easily plot it later on.
    # The prefix is a prefix that we give to node names so that we
    # guarantee that they will be unique

    (rule,children) = tree

    thisnodename = "p%s"%prefix
    nodes = [(thisnodename,rule)]
    edges = []
    for i,child in enumerate(children):
        # Take over the nodes and edges from the children
        childroot,n,newedges = get_nodes_edges(child,"%i%s"%(i,prefix))
        nodes += n
        edges += newedges
        edges += [(thisnodename,childroot)]

    return thisnodename,nodes,edges





def dot_output(tree):
    # Make a little dot output for the particular tree
    _,nodes,edges = get_nodes_edges(tree)

    outp = ""

    outp += "digraph tree {\n"
    
    for (nodename,nodelabel) in nodes:
        outp += "\t%s [label=\"%s\"]\n"%(nodename,nodelabel)

    for (from_node,to_node) in edges:
        outp += "\t%s -> %s\n"%(from_node,to_node)

    outp+="}\n"
    return outp




def tree_to_pdf(tree,fname):
    # Makes a dot graph and outputs to pdf
    outp = dot_output(tree)
    f = open('.tmp.dot','w')
    f.write(outp)
    f.close()
    
    import subprocess
    subprocess.call(['dot','.tmp.dot','-Tpdf','-o',fname])
    # subprocess.call(['rm','.tmp.dot']) # clean up my mess
    
    return



def tree_to_png(tree,fname):
    # Makes a dot graph and outputs to pdf
    outp = dot_output(tree)
    f = open('.tmp.dot','w')
    f.write(outp)
    f.close()
    
    import subprocess
    subprocess.call(['dot','.tmp.dot','-Tpng','-o',fname])
    # subprocess.call(['rm','.tmp.dot']) # clean up my mess
    
    return



def make_rule_probs(g):
    """Given a grammar with rhss (rhs,isCopy,prob) makes dictionary of log rule probs. 
    Keys are strings built from rule names.
    We use the same method for making keys as is used in the parser in case we want to change it"""
    rule_probs={}
    for (lhs,rhss) in g:
        for (rhs,isCopy,p) in rhss:
            rule_probs[rule2string(lhs,rhs,isCopy)]=np.log(p)
    return rule_probs




