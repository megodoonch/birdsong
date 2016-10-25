import pandas as pd


def rules2triples(ops_fsa):
    rules=[]
    for lhs in ops_fsa:
        for (rhs,e) in ops_fsa[lhs]:
            rules.append(('%s->%s %s'%(lhs,e,rhs),(lhs,rhs,e)))

    return rules


def ops_table(history,ops_fsa):
    rules=rules2triples(ops_fsa)
    tab=[]
    for (rule,(lhs,rhs,e)) in rules:
        thisrule={' rule':rule}
        for i in range(len(history)):
            p=history[i]['fsa'][lhs][(rhs,e)]
            thisrule["p.iteration%03d"%i]=p
        tab.append(thisrule)

    return pd.DataFrame(tab)


def trans2pairs(trans):
    bigrams=[]
    for lhs in trans:
        for rhs in trans[lhs]:
            bigrams.append(('%s,%s'%(lhs,rhs),(lhs,rhs)))

    return bigrams

def trans_probs_table(history,trans):
    bigrams=trans2pairs(trans)
    tab=[]
    for (bi,(lhs,rhs)) in bigrams:
        thisrule={' rule':bi}
        for i in range(len(history)):
            p=history[i]['trans_probs'][lhs][rhs]
            thisrule["p.iteration%i"%i]=p
        tab.append(thisrule)

    return pd.DataFrame(tab)





def ll_corpus_table(history):
    """
    make a pandas dataframe of likelihood of training corpus as we trained 

    If we built an elaborate one, it's already in here; otherwise we calculate it

    Arguments
    history   :  list of dicts including trans_probs, fsa, and if elaborate, train_ll

    Returns
    dataframe of LLs over the iterations
    """

    tab=[]
    for i in range(len(history)):
        if 'train_ll' in history[i]:
            ll=history[i]['train_ll']
        else:
            ll=em.ll_corpus(parsed_corpus,history[i]['trans_probs'],history[i]['fsa'])

        this_iter={'iteration':i,
                   'likelihood':ll}
        tab.append(this_iter)

    return pd.DataFrame(tab)
        

def p_parses_table(parsed_corpus,history):
    tab=[]
    
    for i,(s,parse) in enumerate(parsed_corpus):
        this_parse={' sentence':("%i: %s"%(s,' '.join(parse['s']))),'bigrams':' '.join(parse['bis']),'Q':' '.join(parse['rt'][0]),'E':' '.join(parse['rt'][1])}
        for iter in range(1,len(history)):
            this_parse['iteration_%i p'%iter]=history[iter]['parse_ps'][i]
        tab.append(this_parse)
    return pd.DataFrame(tab)



def ll_table(history):
    """
    makes a pandas dataframe of the history of the LLs of both the training and testing corpora.
    only works if we built an elaborate history.

    history : list of dicts including 'train_ll' and 'test_ll'

    Returns
    dataframe of LLs over the iterations
    """
    tab=[]
    for i in range(len(history)):
        this_parse={}
        this_parse['iteration']=i
        this_parse['train LL']=history[i]['train_ll']
        this_parse['test LL']=history[i]['test_ll']
        tab.append(this_parse)

    return pd.DataFrame(tab)
