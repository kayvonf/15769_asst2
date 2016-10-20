CXX ?= g++

HALIDE_PATH = /afs/cs/academic/class/15769-f16/assignments/Halide
CUDA_PATH = /usr/local/cuda
CXXFLAGS += -O3 -g -Wall -std=c++11 -fopenmp

HALIDE_INC += -I$(HALIDE_PATH)/include -I$(HALIDE_PATH)/tools
CUDA_INC += -I$(CUDA_PATH)/include

LDFLAGS += -lHalide -L$(HALIDE_PATH)/bin -lpng

all: inference training

ModelIO.o: ModelIO.h ModelIO.cpp
	$(CXX) $(CXXFLAGS) ModelIO.cpp -c $(HALIDE_INC) $(CUDA_INC)

DataLoaders.o: DataLoaders.h DataLoaders.cpp
	$(CXX) $(CXXFLAGS) DataLoaders.cpp -c $(HALIDE_INC) $(CUDA_INC)

inference: Inference.cpp DataLoaders.o ModelIO.o NetworkDefinitions.h Layers.h
	$(CXX) $(CXXFLAGS) Inference.cpp DataLoaders.o ModelIO.o $(HALIDE_INC) $(LDFLAGS) -o inference

training: Training.cpp DataLoaders.o ModelIO.o NetworkDefinitions.h Layers.h
	$(CXX) $(CXXFLAGS) Training.cpp DataLoaders.o ModelIO.o $(HALIDE_INC) $(LDFLAGS) -o training

clean:
	rm -rf ModelIO.o DataLoaders.o inference training
