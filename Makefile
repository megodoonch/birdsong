default: showdoc

doc: readme.html

showdoc: readme.html
	xdg-open readme.html

readme.html: readme.md
	markdown readme.md > readme.html


clean:
	rm -f readme.html

