PDF=pdflatex -halt-on-error -file-line-error -draftmode
PDFFINAL=pdflatex -halt-on-error -file-line-error --synctex=1

all : clean manual

manual:
	$(PDF) $@
	bibtex manual
	$(PDF) $@
	$(PDFFINAL) $@


clean:
	rm -f *bbl *blg *nls *aux *log *gls *glg *glo *lof
