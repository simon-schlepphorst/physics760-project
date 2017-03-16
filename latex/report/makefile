SHELL = /bin/bash

all_rst := $(wildcard *.tex)
all_pdf := $(all_rst:%.tex=_build/%.pdf)

all_ps := $(wildcard figures/*.ps)
all_fig := $(all_ps:figures/%.ps=_build/%.pdf)

.PRECIOUS: $(all_fig)

all: $(all_pdf)

_build:
	mkdir -p _build

_build/%.pdf: %.tex $(all_fig)
	latexmk -gg -pdflatex='pdflatex -halt-on-error -interaction=batchmode' -pdf -output-directory='_build' $<

_build/%.pdf: figures/%.ps _build
	ps2pdf -sPAPERSIZE=a4 "$<" "$@"

clean:
	$(RM) -r _build
