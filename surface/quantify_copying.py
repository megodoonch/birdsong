

def find_repetitions(sentence):
    copies = []
    for i in range(len(sentence)-1):                   # loop over position in sentence
    	for n in range(1,((len(sentence)-i)//2)+1):     # loop over substrings
    	    if sentence[i:i+n]==sentence[i+n:i+(2*n)]: # if we detect a copy...
                copies.append( (i,n) )
    return copies




def corpus(corpus):

    all_copies = []
    for sentence in corpus:
        # First, establish all copying
        copies = find_repetitions(sentence)
        all_copies += copies

    n_copies = len(all_copies)
    copied_total = sum([ n for (_,n) in all_copies ])

    n_corpus_words = sum([ len(s) for s in corpus ])

    corpus_strings = [ ".".join(s) for s in corpus ]

    return {
        # Some general quantities about the corpus
        "n.unique.sentences"     :len(set(corpus_strings)),
        "n.sentences"            :len(corpus),
        "n.corpus.words"         :n_corpus_words,

        # The 
        "n.repetitions"          :n_copies,
        "total.repeated.material":copied_total,
        "words.per.repeat"       :copied_total/float(n_copies),
        "repetitions.per.word"   :copied_total/float(n_corpus_words),
    }

