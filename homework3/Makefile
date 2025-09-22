CXX 	 = nvcc
CXXFLAGS = -O3 -std=c++11

# 89 for RTX 4060
ARCH 	 = 89
GENCODE  = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

SRC1	 = main_q1.cu
SRC2	 = main_q2.cu
INC	     = .

SRC 	 := $(wildcard *.cpp) $(wildcard *.cu)
EXE 	 := $(patsubst %.cpp, %, $(filter %.cpp, $(SRC))) $(patsubst %.cu, %, $(filter %.cu, $(SRC)))

.PHONY: all main_q1 main_q2 run1 run2 clean

all: $(EXE)

main_q1: $(SRC1)
	$(CXX) $(CXXFLAGS) $(GENCODE) -I$(INC) -o $@ $<

main_q2: $(SRC2)
	$(CXX) $(CXXFLAGS) $(GENCODE) -I$(INC) -o $@ $<

run1: main_q1
	./main_q1

run2: main_q2
	 ./main_q2

clean:
	rm -f *.o *~ *~ main_q1 main_q2
	rm -rf *.dSYM
