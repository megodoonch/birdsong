default: showdoc


doc: readme.html

showdoc: readme.html
	xdg-open readme.html

readme.html: readme.md misc/github-pandoc.css dot
	pandoc readme.md --toc -o readme.html -s -S -H misc/github-pandoc.css

dot: markhov/ops.dot markhov/bi_ops.dot example.dot
	dot -Tpng -o ops.png markhov/ops.dot
	dot -Tpng -o bi_ops.png markhov/bi_ops.dot
	dot -Tpng -o example.png example.dot

latex: conceptual_smart_stuff.pdf
	xdg-open conceptual_smart_stuff.pdf

conceptual_smart_stuff.pdf: conceptual_smart_stuff.tex
	pdflatex conceptual_smart_stuff.tex

clean:
	rm -f readme.html conceptual_smart_stuff.pdf


