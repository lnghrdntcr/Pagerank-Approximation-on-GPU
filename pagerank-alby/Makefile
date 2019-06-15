CUB_PATH=/usr/local/include

ifndef CUB_PATH
$(error CUB_PATH should be set to where "cub" is located (e.g. ~/cub-1.8.0))
endif


FLAGS = -std=c++11 -O3 -use_fast_math -arch=sm_60 -Xcompiler -fopenmp -Xcompiler="-pthread" --default-stream per-thread

all:
	nvcc -I$(CUB_PATH) pagerank_pipeline.cu pagerank.cu main.cpp $(FLAGS) -o pagerank_experiment 
