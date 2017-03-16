SHELL = /bin/bash

all_rst := $(wildcard *.tex)
all_pdf := $(all_rst:%.tex=_build/%.pdf)

all: $(all_pdf)

_build/%.pdf: %.tex
	latexmk -gg -pdflatex='pdflatex -halt-on-error -interaction=batchmode' -pdf -output-directory='_build' $<

clean:
	$(RM) -r _build
