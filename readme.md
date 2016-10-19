# Meaghan & Floris Birdsong Project


## File structure
* `markhov/` contains a model with two Markhov chains: one for the bigrams and one for the operations
	* `markhov.py` houses the main functions for parsing the surface sentences
	* `em.py` has the expectation maximisation learner
	* depricated: `cky.py` has a CKY parser for the regular grammar of the operations. It also has the beginnings of the I/O learner: outside probabilities and the expected counts of a rule in a sentence.
	* `development.py` has some old functions that I don't want to throw away just yet

* Depricated: `ckypy/` contains the basic CKY parser
	* `input` contains output from the previous OCaml script so that we can easily extract the rules
	* `read_cath8_parses.py` reads `input` and does something with it.
	* `cky.py` is the parser we'll probably use. It treats copying as a string operation.
	* `cky_constituent_copy.py` this is a version of the parser that only copies constituents and does so with rules marked as copy rules. Currently imported in `read_cath8_parses.py`.
	* `common.py` contains functions common to both parsers
* Depricated: `mg/` contains a minimalist-style parser
	* `parser.py` is a draft of a minimalist grammar style cky parser that handles copies and transitions. This might be what we want. More later...
	  

## terminology

* *surface string*: the bird's actual sentence, made up of alphabet words
* *Parse string*: sequence of operations. On analogy with *parse tree*.
* *Bigram string*: sequence of words from alphabet. We're distinguishing the surface string from this sequence because the bigrams used without copying are different from the ones used with copying. For example, if *aa* is parsed by `mg mg end` the bigram string is *(open bracket) a a*. If it's parsed by `mg copy end`, the bigram string is *(open bracket) a*.

* *bigram (markhov) chain/automaton*  the markhov chain that generates/accepts the bigram strings and surface strings of alphabet words

* *operations automaton* the probabilistic finite state automaton that generates/accepts the parse strings of operations

* *operations PSG*  The phrase structure grammar equivalent of the operations automaton

## Markhov model

This grammar works as follows: there are four operations, Merge, Copy, Clear, and End. As we build the sentence we also build a buffer which is a suffix of the sentence so far. Merge adds a word to both sentence and buffer. Copy appends the buffer to both the sentence and the buffer itself, and Clear clears the buffer. End ends the sentence.  The transition from one operation to another has a transitional probability and so does the transition from one word to the next.

To reduce clutter, Clear can't reccur unless a Copy intervenes. We can't prevent the parser from including parses in which we clear pointlessly at the end, but we remove them from consideration in `check_ops`. Ultimately we want only parses in which we clear the buffer to reset it to create a new copy and then we actually create that copy.

The main functions in `markhov.py` are:

* `generate_ops` generates a legal sequence of operations using the porbabilities in the ops PFSM
* `generate_string` generates a sentence based on a sequence of operations and the probabilities in the bigrams markhov chain
* `gen` is a wrapper function that generates a string in the language
* `gen_corpus` generates a corpus of length `n`
* `parse` finds for a sentence all the possible parses. A parse is a pair of the sequence of bigrams and the sequence of operations used
* `check_bigrams` calculates the probability of a bigram sequence
* `check_ops` calculates the probability of an operation sequence
* `prob_parse` finds the probability of a single parse by multiplying the probabilities of the bigram sequence and the operation sequence
* `prob_string` calculates the total probability of a string by adding the probabilities of the parses 


### operations

The operations form a probabilistic finite state automaton (PFSA). The edge labels encode the operations Merge, Copy, Clear, and End. The states keep track of things we need to know in order not to generate a bunch of crap like clearing an already empty buffer.

There is also a no-copy grammar operation PFSA.

In the bigram grammar PFSA there are only two states:
*  S : start state
*  F : final state
You can transition from S to F only on the operation End. Merge is a loop back to S.

![operations in no-copy grammar](markhov/bi_ops.png)


In the Copy grammar PFSA the states are as follows:

* S:    this is the start state. From here you must Merge
* COPY: this is a bit of misnomer but it means that we haven't Cleared since the last time we Copied or we haven't cleared yet. From here you can do anything, and only from here can you end. This prevents pointless Clears at the end of the sequence
* CLEAR_S : this is like the start state but for the buffer. This is exactly for clearing. From here you must Merge, which takes you to CLEAR
* CLEAR : from here you can Merge and Copy. YOu can't clear because that makes too many damn parses.

![operations in copy grammar](markhov/ops.png)

For the CKY parser we make an operations phrase structure grammar  in Chomsky Normal Form:

* S->MG NotCL
* NotCL -> MG NotCL
* NotCL -> COPY NotCL
* NotCL -> CLEAR CL_S
* NotCL -> end
* CL_S -> MG CL
* CL -> MG CL
* CL -> COPY NotCL
* MG -> mg
* COPY -> copy
* CLEAR -> clear




### bigrams

Bigrams form a Markhov chain.

When we parse we just keep track of the bigram transitions we actually followed by Merging. So if there's copying in a parse, the bigrams for that parse will just include the copied material once.

It's a little confusing because we store the bigrams used not as a list of pairs but just a substring of the actual sentence.

The bigrams include edge symbols. `parse` in `markhov.py` adds the edge symbols itself. This might need to be changed depending on exactly how we generate the sentences to feed to `parse`

### properties

This grammar only copies embedded copies. No copying indefinite material behind a Copy head like in the old OCaml version. We think this is good: it makes it seem more grammatical and less extra-grammatical.

We have a 2-step grammar. Ops generates a derivation "tree" (really a sequence) and then we generate the sentence from that and the bigram probabilities.


**Something might be off about the ending transitions. Should we really be encoding them with both the edge marker and the end operation?**

I think I've decided not ignore the final transitional probability at least for now. I did make two generators, one that cares and one that doesn't. The parser works the same regardless. I think.

## Learning

We use an expectation-maximisation algorithm to learn the optimal probabilities for the two automata.

For each transition, we take the number of times we took that transition and divide it by the number of times we visited its origin state. The number of times a transition was taken or a state visited is multiplied by the probability of the parse in question. The fraction is the new probability for that transition

![Notes on learner](conceptual_smart_stuff.pdf)




<!-- ### CKY parsing -->

<!-- The CKY parser parses the parse strings. We do this so we can use the Inside-Outside algorithm to train the rule probabilities. This is in `markhov/cky.py`. -->

<!-- Since the grammar is regular, we can simplify quite a bit. We only ever fill the diagonal and the last column, and we always know where to look for a potential sister and where to look in the backpointers. We don't need to loop over partitions, and we only have to fill the last column. -->

<!-- ### Inside-Outside Algorithm -->

<!-- `parse`: We CKY-parse each parse string separately, and when we do the I/O counts we add up all rule uses in all parses of a given surface string. -->

<!-- For example, if the bird says *aa*, `parse` in `markhov.py` will give us two parses: (*mg mg end* , *(aa* ) and (*mg copy end* , *(a* ). We run the CKY parser on each of these. When we calculate the expected counts of the rule `S->MG NotCL` in sentence *aa*, we include the use in both parses. -->

<!-- `outside`: Outside probabilities are currently calculated normally EXCEPT that the top right corner is not initialised to 1, but rather to the proportion of the sentence probabilitiy the parse represents. This might not be necessary: I think this fraction might factor out so that we can just include it in the expected counts calculation for the parse. -->

<!-- `c_phi`: Expected counts are calculated differntly rom usual because the parses are not all housed in the same chart. I'll write more about this tomorrow. -->


<!-- **Development notes** -->

<!-- I tried to put all parse strings into the same cky chart. This doesn't work. The cky parser predicts strings that are not in the original set of parse strings. -->



## Plan

## Misc notes



## Deep thinking

* To some extent, our estimate of how much copying is going on depends on where we place sentence boundaries. Imagine that two identical "sentences" are classified as two sentences, then indeed there is no copying necessary. However, if the two were treated as a single sentence then you need copying. (This is something that Ed thinks about too).


* We're worried about the difference between copying in the grammar and extra-grammatical copying where the bird just repeats himself sometimes. Is there a way to distinguish these? Floris suggests that the cky.py implementation sounds more like extra-linguistic copying. Maybe we want to claim tht the bird's copies are necessarily embedded in each other.

	* I'm not even sure anymore that the way we're copying in even mildly context senstitive, since you don't actually build the things that get copied. You go right into the string and copy that.
