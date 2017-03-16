SHELL = /bin/bash

all_rst := $(wildcard *.tex)
all_pdf := $(all_rst:%.tex=_build/%.pdf)

all_ps := $(wildcard figures/*.ps)
all_fig := $(all_ps:figures/%.ps=_build/%.pdf)

all: $(all_pdf)

_build:
	mkdir -p _build

_build/%.pdf: %.tex $(all_fig)
	latexmk -gg -pdflatex='pdflatex -halt-on-error -interaction=batchmode' -pdf -output-directory='_build' $<

figures/%.ps: _build/%.pdf _build
	ps2pdf $< $@

clean:
	$(RM) -r _build
