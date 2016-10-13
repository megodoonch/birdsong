default: showdoc

dot: ops.dot bi_ops.dot
	dot -Tpdf -o ops.pdf ops.dot
	dot -Tpdf -o bi_ops bi_ops.dot

doc: readme.html

showdoc: readme.html
	xdg-open readme.html

readme.html: readme.md misc/github-pandoc.css
	pandoc readme.md --toc -o readme.html -s -S -H misc/github-pandoc.css

clean:
	rm -f readme.html


