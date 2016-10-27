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


def rule2columns(rule,lhs,rhs,history,grammar,automaton,w,run,begin=None,end=None):
    """
    Given a rule and a bunch of info about run, windows, grammar, etc, 
    builds a dict for a dataframe for the rule
    Dataframe includes how rule probability changed during training, 
    for each run of Windows
    for each training window

    

    """

    if begin==None:
        begin=0
    if end==None:
        end=len(history)

    rule_set=[]
    
    for i in range(begin,end):
        this_rule={' rule':rule}    
        if i < len(history):
            p=history[i][automaton][lhs][rhs]
        else: 
            p=history[begin-1][automaton][lhs][rhs]
        this_rule["prob"]=p
        this_rule['run']=run
        this_rule['training window']=w
        this_rule['iteration']=i
        this_rule['grammar']=grammar
        if automaton == 'fsa':
            this_rule['automaton']='operations'
        elif automaton == 'trans_probs':
            this_rule['automaton']='transitions'
        rule_set.append(this_rule)

    return rule_set




def rule_probs_table(windows,trans,ops_c,ops_nc,fill=False):
    """
    Makes a dataframe of the results of running Windows, potentially multiple times
    
    Arguments
    window : list of list of pairs ((ll_copy,history_copy),(ll_no_copy,history_no_copy))
    
    Returns
    
    """
    c_rules=rules2triples(ops_c)
    nc_rules=rules2triples(ops_nc)
    bigrams=trans2pairs(trans)
    tab=[]
    # for each complete run of windows (re-initialised)
    for run,window in enumerate(windows):
        #print (run)
        #for each window:
        for w,((_,history_copy),(_,history_no_copy)) in enumerate(window):

            # operations FSAs

            # copy grammar
            for rule,(lhs,rhs,e) in c_rules:

                thisrule=rule2columns(rule,lhs,(rhs,e),history_copy,'copy','fsa',w,run)
                tab+=thisrule

            # no copy grammar
            nc=len(history_no_copy)
            for rule,(lhs,rhs,e) in nc_rules:
                thisrule=rule2columns(rule,lhs,(rhs,e),history_no_copy,'no copy','fsa',w,run)
                tab+=thisrule

                if fill:
                    for i in range (nc,len(history_copy)):
                        thisrule=rule2columns(rule,lhs,(rhs,e),history_no_copy,'no copy','fsa',w,run,nc,len(history_copy))
                        tab+=thisrule

            # transitional probabilities
            for rule,(lhs,rhs) in bigrams:
                # copy grammar
                thisrule=rule2columns(rule,lhs,rhs,history_copy,'copy','trans_probs',w,run)
                tab+=thisrule
                # no copy grammar
                thisrule=rule2columns(rule,lhs,rhs,history_no_copy,'no copy','trans_probs',w,run)
                tab+=thisrule
                
    for i in range(len(tab)):
        assert type(tab[i])==dict, "element %i had type %s"%(i,type(tab[i]))
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
        this_iter={}
        this_iter['iteration']=i
        this_iter['train LL']=history[i]['train_ll']
        this_iter['test LL']=history[i]['test_ll']
        tab.append(this_iter)

    return pd.DataFrame(tab)



def ll_window(windows,fill=False):
    """
    Makes a dataframe of the results of running Windows, potentially multiple times
    
    Arguments
    window : list of list of pairs ((ll_copy,history_copy),(ll_no_copy,history_no_copy))
    
    Returns
    
    """
    tab=[]
    # for each complete run of windows (re-initialised)
    for run,window in enumerate(windows):
        #print (run)
        #for each window:
        for w,((_,history_copy),(_,history_no_copy)) in enumerate(window):
            #print (w)
            # copy grammar
            #print ('copy')
            for i in range(len(history_copy)):
                #print (i)
                # for each iteration of EM
                this_iter={}
                this_iter['run']=run
                this_iter['training window']=w
                this_iter['iteration']=i
                this_iter['grammar']='copy'
                this_iter['train LL']=history_copy[i]['train_ll']
                this_iter['test LL']=history_copy[i]['test_ll']
                tab.append(this_iter)
            # no copy grammar
            nc=len(history_no_copy)
            for i in range(nc):
                # for each iteration of EM
                this_iter={}
                this_iter['run']=run
                this_iter['training window']=w
                this_iter['iteration']=i
                this_iter['grammar']='no copy'
                this_iter['train LL']=history_no_copy[i]['train_ll']
                this_iter['test LL']=history_no_copy[i]['test_ll']
                tab.append(this_iter)
            if fill:
                for i in range (nc,len(history_copy)):
                    this_iter={}
                    this_iter['run']=run
                    this_iter['training window']=w
                    this_iter['iteration']=i
                    this_iter['grammar']='no copy'
                    this_iter['train LL']=history_no_copy[nc-1]['train_ll']
                    this_iter['test LL']=history_no_copy[nc-1]['test_ll']
                    tab.append(this_iter)
                   

    return pd.DataFrame(tab)



