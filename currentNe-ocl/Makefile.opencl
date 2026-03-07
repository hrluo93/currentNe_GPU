CXX ?= g++
UNAME_S := $(shell uname -s)
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

ifeq ($(UNAME_S),Darwin)
OPENMP_CXXFLAGS ?=
OPENMP_LDFLAGS ?=
PIE_CXXFLAGS ?=
PIE_LDFLAGS ?=
else
OPENMP_CXXFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp
PIE_CXXFLAGS ?= -fno-pie -mcmodel=large
PIE_LDFLAGS ?= -Xlinker -no-pie
endif

CPPFLAGS += -DUSE_OPENCL_NE
CXXFLAGS += -O3 -std=gnu++17 $(OPENMP_CXXFLAGS) -I$(ROOT) $(PIE_CXXFLAGS)
LDFLAGS += $(OPENMP_LDFLAGS) $(PIE_LDFLAGS)

ifeq ($(UNAME_S),Darwin)
LDLIBS += -framework OpenCL
else
LDLIBS += -lOpenCL
endif

TARGET ?= $(ROOT)/currentNe_ocl
OBJS := $(ROOT)/currentne-cpu.o $(ROOT)/opencl_ld.o $(ROOT)/progress.o

all: $(TARGET)

$(ROOT)/currentne-cpu.o: $(ROOT)/currentne-cpu.cpp $(ROOT)/opencl_ld.hpp $(ROOT)/lib/progress.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT)/currentne-cpu.cpp -o $@

$(ROOT)/opencl_ld.o: $(ROOT)/opencl_ld.cpp $(ROOT)/opencl_ld.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT)/opencl_ld.cpp -o $@

$(ROOT)/progress.o: $(ROOT)/lib/progress.cpp $(ROOT)/lib/progress.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT)/lib/progress.cpp -o $@

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJS)
