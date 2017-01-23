all: html txt

html: *.py
	pydoc3 -w ./*.py

txt: *.py
	pydoc3 ./*.py > vec4ir-doc.txt

clean:
	rm -f *.html
	rm -f vec4ir-doc.txt
