Calcula el area de una superficie a partir de una nube de puntos.

### Versión definitiva (OpenCL y CUDA)
El código que implementa openCL y CUDA se encuentra dentro del directorio **`montecarlo_openCL/`**

### Versiones anteriores
- Compilar versión de CPU con openMP: `gcc -o area area.c -lSDL2 -lm -fopenmp && ./area`
- Compilar versión de GPU con CUDA: `nvcc -o areacuda area.cu -lSDL2 && ./areacuda`
