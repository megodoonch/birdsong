# Meaghan & Floris Birdsong Project


## File structure
* `ckypy/` contains the basic CKY parser
	* `input` contains output from the previous OCaml script so that we can easily extract the rules
	* `read_cath8_parses.py` reads `input` and does something with it.
	* `cky.py` is the parser we'll probably use. It treats copying as a string operation.
	* `cky_constituent_copy.py` this is a version of the parser that only copies constituents and does so with rules marked as copy rules. Currently imported in `read_cath8_parses.py`.
	* `common.py` contains functions common to both parsers
* `mg/` contains a draft of a minimalist-style parser
	* `parser.py` is a draft of a minimalist grammar style cky parser that handles copies and transitions. This might be what we want. More later...


## Plan
* Make a CKY parser.
  * I'm satisfied the parser is correct now, and the probability calculation too. 
    * Tested with Catalan numbers. Both parsers generate the correct number of parses and the same probabilities when they use the same grammar. This is just for non-copy grammars.
Catalan numbers show up a lot in counting recursive things. From Wikipedia:

	- Cn is the number of different ways n + 1 factors can be completely parenthesized (or the number of ways of associating n applications of a binary operator). For n = 3, for example, we have the following five different parenthesizations of four factors:

			   ((ab)c)d     (a(bc))d     (ab)(cd)     a((bc)d)     a(b(cd))


	- Successive applications of a binary operator can be represented in terms of a full binary tree. (A rooted binary tree is full if every vertex has either two children or no children.) It follows that Cn is the number of full binary trees with n + 1 leaves


* Add outside probabilities
* ...


## Misc notes

* General difference in parser design between OCaml one and this Python one: M's Ocaml code puts all the functionality (up to and including outside probabilities) into the same CKY chart, while F makes a bunch of separate charts
* we're not sure if F's approach is better or not.


## Deep thinking

* To some extent, our estimate of how much copying is going on depends on where we place sentence boundaries. Imagine that two identical "sentences" are classified as two sentences, then indeed there is no copying necessary. However, if the two were treated as a single sentence then you need copying. (This is something that Ed thinks about too).


* We're worried about the difference between copying in the grammar and extra-grammatical copying where the bird just repeats himself sometimes. Is there a way to distinguish these? Floris suggests that the cky.py implementation sounds more like extra-linguistic copying. Maybe we want to claim tht the bird's copies are necessarily embedded in each other.

	* I'm not even sure anymore that the way we're copying in even mildly context senstitive, since you don't actually build the things that get copied. You go right into the string and copy that.
