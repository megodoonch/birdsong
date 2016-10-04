# Meaghan & Floris Birdsong Project


## File structure
* `ckypy` contains the basic CKY parser
  * `input` contains output from the previous OCaml script so that we can easily extract the rules
  * `read_cath8_parses.py` reads `input` and does something with it.
  * `ckypy.py` this is the parser proper, imported in `read_cath8_parses.py`.
  


## Plan
* Make a CKY parser.
* ...


## Misc notes

* Main difference in parser design between M&F: M's Ocaml code puts all the functionality (up to and including outside probabilities) into the same CKY chart, while F makes a bunch of separate charts
* we're not sure if F's approach is better or not.
