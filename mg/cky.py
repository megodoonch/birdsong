"""
This is a first stab an an MG-style CKY-style parser for bigrams plus copying

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





## names of dictionary keys for cky chart

TR = "Tr"
CP="Cp"
BP="BP"
PR="P"

def parse(sentence,trans):
    """
    parses based on transitions plus copy
    probs don't work right yet -- might just be the copies which i haven't dealt with yet
    """

    
    n=len(sentence)
    
    chart = [ [ {TR:[], CP:[], BP:[], PR:None} for _ in range(n+1) ] for _ in range(n) ]

    for j in range(1,n+1): ## loop over columns, fill them from the bottom up
        chart[j-1][j][TR]=(sentence[j-1],sentence[j-1]) #starts and ends with itself
        chart[j-1][j][BP].append((sentence[j-1],"lex")) # backpointer (word,lex)
        chart[j-1][j][PR]=0. # this might not be right
        
                
        #check if the word is a copy of the following word
        if j<n and sentence[j]==sentence[j-1]:
            chart[j-1][j][CP].append("-copy") # mark the early copy
            chart[j][j+1][CP].append("+copy") # mark the late copy
            # print (j-1,j)
            # print sentence[j]
            # print sentence[j-1]
            #chart[j-1][j][BP].append("copy")

        # Loop over rows, backwards (bottom-up)
        for i in range(j-2,-1,-1): # we start at j-2 because we already did the diagonal
            copy_end = 2*j-i
            if copy_end <= n+1: # if we've got room for a copy
                # print (i,j)
                # print sentence[i:j]
                # print sentence[j:(2*j-i)]
                if sentence[i:j]==sentence[j:copy_end]:
                    chart[i][j][CP].append("-copy") #mark the early copy
                    chart[j][copy_end][CP].append("+copy") #mark the late copy
                    #chart[i][j][BP].append("copy")

            for k in range(i+1,j): # loop over partitions
                # Check whether we have the constituents previously recognised
                left,right = chart[i][k],chart[k][j]
                if len(left[TR])>0 and len(right[TR])>0: # only look if we actually have daughters
                    (a,b), (c,d) = left[TR], right[TR] # the TRansitions
                    if c in trans[b]:                # check the grammar for the transition between the sisters. b is the last word of the left daughter and c is the first of the right.
                        if (a,d) not in chart[i][j][TR]: #only write it once
                            chart[i][j][TR]=(left[TR][0],right[TR][1]) #the new pair is the outer elements of the combined string
                        print(i,j)
                        print(b,c)
                        print ("trans p ",trans[b][c])
                        print ("L p ",left[PR])
                        print ("R p ",right[PR])
                        print ("mult ", trans[b][c]+left[PR]+right[PR])
                        print ("existing ",chart[i][j][PR])
                        print ("sum ",log_add(chart[i][j][PR],
                                trans[b][c]+left[PR]+right[PR]))
                        chart[i][j][PR]= log_add(chart[i][j][PR],
                                                   trans[b][c]+left[PR]+right[PR]) # this might not be right
                        chart[i][j][BP].append((k,"merge")) # rule and partition in backpointers
                    # print i,j,k
                    # print left[CP]
                    # print right[CP]
                    if k-i==j-k and "-copy" in left[CP] and "+copy" in right[CP]: #make sure both daughters are marked as matching, only if they're actually the same length.
                        if (a,d) not in chart[i][j][TR]: #only write it once
                            chart[i][j][TR]=(left[TR][0],right[TR][1]) #this does belong in the transitions
                        chart[i][j][PR] = log_add(chart[i][j][PR],
                                                    right[PR])
                        chart[i][j][BP].append((k,"copy")) # this is a copy operation, not merge

    return chart



def collect_trees(chart,sentence,transition=("[","]"),from_i=0,to_i=None):
    """
    Once we've parsed the sentence we can collect the trees.

    Arguments:
    chart      :  the CKY chart
    sentence   : string list
    from_i     : int -- start of the span you're interested in. Default 0 (start of sent)
    to_i       : int -- end of the span you're interested in. Default len(sent) (end of sent)

    output     : list of trees. Trees are tuples of (rule used, list of daughter trees)
    """
    
    if to_i==None:
        to_i=len(sentence)

    assert(chart[from_i][to_i][TR]==transition)

    def reconstruct(i,j):
        # GIVE ALL RECONSTRUCTIONS OF THE category between i and j, using back pointers 
        # (this will be called recursively.
        # We do this "inner function" kind of construction so that we don't need to keep
        # passing along copies of the backpointers and chart

        #print "Reconstructing (%i,%i)"%(i,j)

        reconstructions = []

        # For each of the reconstructions, collect the trees
        for (k,rule) in chart[i][j][BP]:

            # for lexical items
            if rule=="lex":
                reconstructions.append( (k,[]) )

            elif rule=="copy":
                trees_right = reconstruct(k,j)
                for tree in trees_right:
                    reconstructions.append( (rule, [tree] ) )
                
            elif rule=="merge":
                # First let's reconstruct each of the subtrees
                trees_rhsB = reconstruct(i,k)
                trees_rhsC = reconstruct(k,j)

                # Ok, then we should combine all reconstructions from both sides of the rule
                for reconstrB in trees_rhsB:
                    for reconstrC in trees_rhsC:

                        # And then put them together
                        reconstruction = (rule,  
                                          [reconstrB,
                                           reconstrC])
                        reconstructions.append( reconstruction )
                    
        return reconstructions
                        
    return reconstruct(from_i,to_i)
                        


##### PRINTERS #####

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
            if ch[i][j][TR]!=[] or ch[i][j][CP]!=[]:
                print ("(%i,%i)"%(i,j),ch[i][j])
    print ("### end Chart ###")




