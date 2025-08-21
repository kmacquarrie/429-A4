TARGET_ISA=x86

CFLAGS=-I$(GEM_PATH)/include

LDFLAGS=-L$(GEM_PATH)/util/m5/build/$(TARGET_ISA)/out -lm5

OBJECTS=GeMM
OBJECT_BIG=GeMMBig

CXX=clang++

GeMM: GeMM.cc
	$(CXX) -o $(OBJECTS) GeMM.cc $(CFLAGS) $(LDFLAGS) -ggdb

GeMMBig: GeMM_big.cc
	$(CXX) -o $(OBJECT_BIG) GeMM_big.cc $(CFLAGS) $(LDFLAGS) -g

clean:
	rm -f $(OBJECTS)
