#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <SDL2/SDL.h>
#include "kernel_render.h"
#include <time.h>

__device__ int puntoEnPoligono(Punto *polygon, int n, Punto point){
    int winding_number = 0;

    for (int i = 0; i < n; i++){
        Punto p1 = polygon[i];
        Punto p2 = polygon[(i + 1) % n];

        if (p1.y <= point.y) {
            if (p2.y > point.y && ((p2.x - p1.x) * (point.y - p1.y) - (point.x - p1.x) * (p2.y - p1.y)) > 0)
                winding_number++;
        } else {
            if (p2.y <= point.y && ((p2.x - p1.x) * (point.y - p1.y) - (point.x - p1.x) * (p2.y - p1.y)) < 0)
                winding_number--;
        }
    }
    return winding_number != 0;
}

__global__ void texturaKernel(unsigned char *d_texture, int ancho, int alto, Punto *d_polygon,
                            int nvert, Punto *d_points, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints){
        Punto point = d_points[idx];
        int x = point.x;
        int y = point.y;

        int dentro = puntoEnPoligono(d_polygon, nvert, point);
        int texIdx = 4 * ((alto - y -1) * ancho + x);

        if (dentro){
            d_texture[texIdx] = 50;
            d_texture[texIdx + 1] = 200;
            d_texture[texIdx + 2] = 50;
            d_texture[texIdx + 3] = 255;
        } else {
            d_texture[texIdx] = 200;
            d_texture[texIdx + 1] = 50;
            d_texture[texIdx + 2] = 50;
            d_texture[texIdx + 3] = 255;
        }
    }
}

void liberarRecursos(SDL_Window *window, SDL_Renderer *renderer, SDL_Texture *texture, 
                     Punto *h_points, unsigned char *d_texture, Punto *d_polygon, Punto *d_points, 
                     unsigned char *h_texture) {
    if (h_texture) free(h_texture);
    if (h_points) free(h_points);
    if (d_texture) cudaFree(d_texture);
    if (d_polygon) cudaFree(d_polygon);
    if (d_points) cudaFree(d_points);
    if (texture) SDL_DestroyTexture(texture);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

void ejecutarTextura(Punto *polygon, int nvert, const int numPoints, int ancho, int alto){
    clock_t start_func, end_func;
    double ms_func;
    start_func = clock();
    
    Punto *h_points = NULL;
    Punto *d_polygon = NULL;
    Punto *d_points = NULL;
    unsigned char *h_texture = NULL;
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0){
        printf("Error al inicializar SDL\n");
        return;
    }

    SDL_Window *window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, ancho, alto, SDL_WINDOW_SHOWN);
    if (!window){
        printf("Error al crear la ventana\n");
        SDL_Quit();
        return;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer){
        printf("Error al crear el renderizador\n");
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, ancho, alto);
    if (!texture){
        printf("Error al crear la textura\n");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    unsigned char *d_texture;
    if (cudaMalloc((void**)&d_texture, ancho * alto * 4 * sizeof(unsigned char)) != cudaSuccess) {
        printf("Error en cudaMalloc para d_texture\n");
        liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
        return;
    }

    h_points = (Punto*)malloc(numPoints * sizeof(Punto));
    if (!h_points){
        printf("Error en malloc para h_points\n");
        liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
        return;
    }

    for (int i = 0; i < numPoints; i++){
        double x = (double)rand() / RAND_MAX * ancho;
        double y = (double)rand() / RAND_MAX * alto;
        h_points[i].x = (int)x;
        h_points[i].y = (int)y;
    }

    if (cudaMalloc((void**)&d_polygon, nvert * sizeof(Punto)) != cudaSuccess){
        printf("Error en cudaMalloc para d_polygon\n");
        liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
        return;
    }
    cudaMemcpy(d_polygon, polygon, nvert * sizeof(Punto), cudaMemcpyHostToDevice);
    if (cudaMalloc((void**)&d_points, numPoints * sizeof(Punto)) != cudaSuccess) {
        printf("Error en cudaMalloc para d_points\n");
        liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
        return;
    }
    cudaMemcpy(d_points, h_points, numPoints * sizeof(Punto), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock -1) / threadsPerBlock;

    printf("Hilos por bloque: %d\nBloques por grid: %d\n", threadsPerBlock, blocksPerGrid);

    clock_t start_kernel, end_kernel;
    double ms_kernel;
    start_kernel = clock();
    
    texturaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_texture, ancho, alto, d_polygon, nvert, d_points, numPoints);

    end_kernel = clock();
    ms_kernel = (double)(end_kernel - start_kernel) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución del kernel texturaKernel: %.3f ms\n", ms_kernel * 1000);

    h_texture = (unsigned char *)malloc(ancho * alto * 4 * sizeof(unsigned char));
    if (!h_texture){
        printf("Error en malloc para h_texture\n");
        liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
        return;
    }
    int quit = 0;
    SDL_Event e;
    end_func = clock();
    ms_func = (double)(end_func - start_func) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución del ejecutarTextura: %.3f ms\n", ms_func * 1000);
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }

        cudaMemcpy(h_texture, d_texture, ancho * alto * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        SDL_UpdateTexture(texture, NULL, h_texture, ancho * 4);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    liberarRecursos(window, renderer, texture, h_points, d_texture, d_polygon, d_points, h_texture);
}
