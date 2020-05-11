
# fdemoullin
# generate this makefile instead of using cmake to run on GPGPU-Sim

PROJECT_NAME = CWBVH_make
BUILD_DIR = ./
SRC_DIR = ../src/

# OptiX
USING_OPTIX=0# 1 enables OptiX
OPTIX_INCLUDE_DIR = ~/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/include
OPTIX_LIB = /home/francois/Documents/RayTracing/NVIDIA-OptiX-SDK-5.1.0-linux64/lib64/ # make sure this is in LD_LIBRARY_PATH

# Embree3
EMBREE_INCLUDE_DIR = ~/Documents/RayTracing/embree/include
EMBREE_LIB = ~/Documents/RayTracing/embree/build

# CUDA
# plain cuda, use this option for running on hardware not on gpgpu-sim
# CUDAPATH = ~/cuda/include
NVCC = nvcc
NVCCFLAGS = -c -g -std=c++11
CUDAARCH = -gencode arch=compute_75,code=compute_75

ifeq "$(USING_OPTIX)" "1"
	LFLAGS = -lcuda --cudart shared -lembree3 -ltbb -loptix_prime -loptix
	DFLAGS = -DOPTIX_PRIME
else
	LFLAGS = -lcuda --cudart shared -lembree3 -ltbb
	DFLAGS = 
endif

GPUDEBUGFLAG = -G # Only use it when debugging the kernel in cuda gdb, don't set this when running with GPGPU-Sim!


all: build

build: gpu cpu
	$(NVCC) $(CUDAARCH) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o -L $(OPTIX_LIB) -L $(EMBREE_LIB) $(LFLAGS)

gpu:
	$(NVCC) --keep $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cu $(DFLAGS)

cpu:			
	$(NVCC) $(NVCCFLAGS) $(CUDAARCH) $(SRC_DIR)*.cpp -I $(OPTIX_INCLUDE_DIR) -I $(EMBREE_INCLUDE_DIR) $(DFLAGS)

clean:
	find . -type f -name _cuobjdump_list_ptx_\* -exec rm {} \;
	find . -type f -name _app_cuda_version_\* -exec rm {} \;
	rm -f *.o
