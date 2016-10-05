# Meaghan & Floris Birdsong Project


## File structure
* `ckypy/` contains the basic CKY parser
	* `input` contains output from the previous OCaml script so that we can easily extract the rules
	* `read_cath8_parses.py` reads `input` and does something with it.
	* `ckypy.py` this is the parser proper, imported in `read_cath8_parses.py`.
  


## Plan
* Make a CKY parser.
  * **Except for the treatment of copies**, I'm satisfied the parser is correct now, and the probability calculation too. 
    * Tested with Catalan numbers:
Catalan numbers show up a lot in counting recursive things. From Wikipedia:

	- Cn is the number of different ways n + 1 factors can be completely parenthesized (or the number of ways of associating n applications of a binary operator). For n = 3, for example, we have the following five different parenthesizations of four factors:

			   ((ab)c)d     (a(bc))d     (ab)(cd)     a((bc)d)     a(b(cd))


	- Successive applications of a binary operator can be represented in terms of a full binary tree. (A rooted binary tree is full if every vertex has either two children or no children.) It follows that Cn is the number of full binary trees with n + 1 leaves

* Fix copies in parser
  * Right now, a copy needs to be buildable as a constituent. This is not how copies work in the grammar I used in the earlier version. Rather, a copy is like a silent head that copies material to its left, even if that material isn't a constituent.
  * the way I did this was to populate the whole chart with copy heads, since a copy could be anywhere. Unlike real silent heads, the copy could span several words, so it doesn't just go on the diagonal
* Make Notebook to illustrate parser
* Add outside probabilities
* ...


## Misc notes

* Main difference in parser design between M&F: M's Ocaml code puts all the functionality (up to and including outside probabilities) into the same CKY chart, while F makes a bunch of separate charts
* we're not sure if F's approach is better or not.


## Deep thinking

* To some extent, our estimate of how much copying is going on depends on where we place sentence boundaries. Imagine that two identical "sentences" are classified as two sentences, then indeed there is no copying necessary. However, if the two were treated as a single sentence then you need copying. (This is something that Ed thinks about too).



