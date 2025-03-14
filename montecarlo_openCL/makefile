CUDA_FLAGS = -arch=sm_75 -Xcompiler="-Wall"
C_FLAGS = -Wall -O2

SDL2_FLAGS = $(shell sdl2-config --cflags --libs)
OPENCL_FLAGS = -lOpenCL

all: montecarlo

montecarlo: kernel_render.o opencl_area.o main.o
	nvcc $(CUDA_FLAGS) -o $@ $^ $(SDL2_FLAGS) $(OPENCL_FLAGS)

%.o: %.cu
	nvcc $(CUDA_FLAGS) -c $< -o $@

%.o: %.c
	gcc $(C_FLAGS) -c $< -o $@

clean:
	rm -f kernel_render.o opencl_area.o main.o montecarlo

run: all
	./montecarlo
