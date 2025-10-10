CXX := /usr/bin/g++
NVCC := /usr/local/cuda/bin/nvcc

# 按你的显卡改：Ada/4090=sm_89，Ampere/30系=sm_80，V100=sm_70
ARCH ?= sm_89

# 关键：-mcmodel=large 以及 no-pie
CXXFLAGS := -O3 -std=c++14 -I. -fopenmp -fno-pie -mcmodel=large
NVCCFLAGS := -O3 -std=c++14 -I. -arch=$(ARCH) -Xcompiler -fopenmp -Xcompiler -fno-pie -Xcompiler -mcmodel=large
LDFLAGS := -Xlinker -no-pie --cudart=shared -lgomp

TARGET := currentNe_gpu
OBJS := gpu_ld.o currentNe.o

all: $(TARGET)

gpu_ld.o: gpu_ld.cu gpu_ld.cuh lib/progress.hpp
	$(NVCC) $(NVCCFLAGS) -c gpu_ld.cu -o $@

# 用 g++ 编译 currentNe.cpp（带 USE_CUDA_NE 宏）
currentNe.o: currentNe.cpp lib/progress.hpp gpu_ld.cuh
	$(CXX) $(CXXFLAGS) -DUSE_CUDA_NE -c currentNe.cpp -o $@

# 用 nvcc 链接（带 no-pie）
$(TARGET): $(OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $(OBJS) -Xcompiler -mcmodel=large $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
