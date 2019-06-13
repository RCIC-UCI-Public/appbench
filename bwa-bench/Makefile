SHELL=/bin/bash
INPUTDATA =testdir.tar
IDATA_HASH = 1Z8RV9qxuTJK2W4hXyQ1a9Z0hrNh-R03I
GGET =  ../gget.sh
SUBDIR = testdir

prep: $(INPUTDATA)
	tar xvf $(INPUTDATA)
	mv $(SUBDIR)/* .

$(INPUTDATA):
	$(GGET) $(IDATA_HASH) $(INPUTDATA)

bench: prep
	NTASKS=8 ./benchit
	

