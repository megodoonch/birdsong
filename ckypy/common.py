"""
Here we put some functions that both parsers need

"""
import numpy as np

def log_add(logx,logy):
    """This adds two log-transformed variables,
    taking care of the underflow that you usually find when you do this

    Arguments: 
    logx : float
    logy : float"""

    if logx==None: # This is just a hack so that I can give a not-defined sum
        return logy
    if logy==None:
        return logx

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


##### PRINTING #######

def rule2string(lhs,rhs,isCopy=False):
    """
    Makes a string from a rule. Used in printing and in making a dict of rules and probs.

    Arguments:
    lhs    : string
    rhs    : string list
    isCopy : boolean indicating a copy rule. Only used in cky_constituent_copy
    """
    s = "%s->%s"%(lhs,".".join(rhs))
    if isCopy:
        s+=".copy"
    return s




def grammar2string(grammar):
    """
    Prints a grammar as a readable string.
    
    Arguments:
    grammar : a simple grammar with no probabilities and no isCopy
    """
    s = ""
    for (lhs,rhss) in grammar:
        for rhs in rhss:
            s+=rule2string(lhs,rhs)+"\n"
    return s






def print_chart(ch):
    """
    Prints any list of lists of lists. 
    For a given cell, prints the coordinates and contents if it's not empty.
    We use this to print the CKY chart. Also works for the backpointers chart.

    Arguments:
    ch : list list list
    """
    print ("### Chart ###")
    for i,row in enumerate(ch):
        for j,col in enumerate(ch[i]):
            if len(ch[i][j])>0:
                print ("(%i,%i)"%(i,j),ch[i][j])
    print ("### end Chart ###")


def print_backpointers(ch):
    g_length = len(ch[0][0])
    print ("### Backpointers ###")
    for i,row in enumerate(ch):
        for j,col in enumerate(ch[i]):
            if any(len(ch[i][j][m])>0 for m in range(g_length)) :
                print ("(%i,%i)"%(i,j),ch[i][j])
    print ("### end Backpointers ###")




####### CHANGE THE GRAMMAR ##########



def make_rule_probs(g,log=False):
    """Given a grammar with rhss (rhs,prob) makes dictionary of log rule probs. 
    Keys are strings built from rule names.
    We use the same method for making keys as is used in the parser in case we want to change it
    If log=False, input probs are not log-transformed.

    Arguments:
    g   : category * (category list * float) list list 
           OR category * (category list * bool * float) list list
    log : Bool indicates whether probs are already log(p)
    """
    rule_probs={}
    for (lhs,rhss) in g:
        for rhs in rhss:
            if len(rhs)==2:
                (rhs,p)=rhs
                if not log: p=np.log(p)
                rule_probs[rule2string(lhs,rhs)]=p
            elif len(rhs)==3:
                (rhs,isCopy,p)=rhs
                if not log: p=np.log(p)
                rule_probs[rule2string(lhs,rhs,isCopy)]=p

    return rule_probs


##### DEALING WITH OUTPUTS ######


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

    outp += "digraph tree {\n node [shape = none; height=0; width=0];\n edge [dir=none];"
    
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




