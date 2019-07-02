SHELL=/bin/bash
INPUTDATA = Wunderlich.tar 
IDATA_HASH = 1VXU064iFiurOAb1xXcaVv4stom4ceSGr 
GGET =  ../gget.sh
SUBDIR = testdir

prep: $(INPUTDATA)
	tar xvf $(INPUTDATA)

$(INPUTDATA):
	$(GGET) $(IDATA_HASH) $(INPUTDATA)

bench: 
	NTASKS=8 ./benchit
	

