default: showdoc

dot: ops.dot bi_ops.dot
	dot -Tpdf -o ops.pdf ops.dot
	dot -Tpdf -o bi_ops bi_ops.dot

doc: readme.html

showdoc: readme.html
	xdg-open readme.html

readme.html: readme.md misc/github-pandoc.css
	pandoc readme.md --toc -o readme.html -s -S -H misc/github-pandoc.css

latex: conceptual_smart_stuff.pdf
	xdg-open conceptual_smart_stuff.pdf

conceptual_smart_stuff.pdf: conceptual_smart_stuff.tex
	pdflatex conceptual_smart_stuff.tex

clean:
	rm -f readme.html conceptual_smart_stuff.pdf


