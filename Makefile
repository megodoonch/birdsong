default: showdoc

doc: readme.html

showdoc: readme.html
	xdg-open readme.html

readme.html: readme.md misc/github-pandoc.css
	pandoc readme.md --toc -o readme.html -s -S -H misc/github-pandoc.css

clean:
	rm -f readme.html

