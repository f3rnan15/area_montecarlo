#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <SDL2/SDL.h>

#define WINDOW_WIDTH 400
#define WINDOW_HEIGHT 400

typedef struct {
    int x, y;
} Punto;

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

__global__ void calculateAreaKernel(Punto *d_polygon, int nvert, Punto *d_points, int *d_results, int numPoints, int ancho, int alto) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints){
        Punto punto = d_points[idx];
        d_results[idx] = puntoEnPoligono(d_polygon, nvert, punto);
    }
}

double calculateArea(Punto *polygon, int nvert, int ancho, int alto, int numPoints, Punto *h_points){
    Punto *d_polygon;
    Punto *d_points;
    int *d_results;

    cudaMalloc((void**)&d_polygon, nvert * sizeof(Punto));
    cudaMalloc((void**)&d_points, numPoints * sizeof(Punto));
    cudaMalloc((void**)&d_results, numPoints * sizeof(int));

    cudaMemcpy(d_points, h_points, numPoints * sizeof(Punto), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polygon, polygon, nvert * sizeof(Punto), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock -1) / threadsPerBlock;

    calculateAreaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_polygon, nvert, d_points, d_results, numPoints, ancho, alto);

    int *h_results = (int*)malloc(numPoints * sizeof(int));
    cudaMemcpy(h_results, d_results, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    int countInside = 0;
    for (int i = 0; i < numPoints; i++) countInside += h_results[i];

    cudaFree(d_polygon);
    cudaFree(d_points);
    cudaFree(d_results);
    free(h_results);

    double area = (double)countInside / numPoints * ancho * alto;
    return area;
}

double calculateRealArea(Punto *polygon, int n) {
    double area = 0;

    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[i].y * polygon[j].x;
    }

    return fabs(area) / 2.0;
}


Punto centroide;

Punto calcularCentroide(Punto *puntos, int n){
    Punto centroide = {0, 0};
    for (int i = 0; i < n; i++){
        centroide.x += puntos[i].x;
        centroide.y += puntos[i].y;
    }
    centroide.x /= n;
    centroide.y /= n;
    return centroide;
}

int compararPorAngulos(const void *a, const void *b){
    Punto *p1 = (Punto *)a;
    Punto *p2 = (Punto *)b;
    double angulo1 = atan2(p1->y - centroide.y, p1->x - centroide.x);
    double angulo2 = atan2(p2->y - centroide.y, p2->x - centroide.x);

    return (angulo1 < angulo2) ? -1 : (angulo1 > angulo2) ? 1 : 0;
}

void ordenarPuntos(Punto *puntos, int n){
    centroide = calcularCentroide(puntos, n);
    qsort(puntos, n, sizeof(Punto), compararPorAngulos);
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



int main() {

    if (SDL_Init(SDL_INIT_VIDEO) < 0){
        printf("Error al inicializar SDL\n");
        return -1;
    }

    SDL_Window *window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window){
        printf("Error al crear la ventana\n");
        SDL_Quit();
        return -1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer){
        printf("Error al crear el renderizador\n");
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!texture){
        printf("Error al crear la textura\n");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    unsigned char *d_texture;
    cudaMalloc((void**)&d_texture, WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(unsigned char));


    Punto polygon[] = {{100, 100}, {200, 200}, {100, 200}, {200, 100}};
    //Punto polygon[] = {{50, 50},{150, 75},{125, 150},{200, 200},{100, 250},{50, 200},{25, 100}};
    int nvert = sizeof(polygon) / sizeof(Punto);

    int ancho = 400;
    int alto = 400;
    int numPoints = 1000000;

    ordenarPuntos(polygon, nvert);

    Punto *h_points = (Punto*)malloc(numPoints * sizeof(Punto));

    for (int i = 0; i < numPoints; i++){
        double x = (double)rand() / RAND_MAX * ancho;
        double y = (double)rand() / RAND_MAX * alto;
        h_points[i].x = (int)x;
        h_points[i].y = (int)y;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    double area = calculateArea(polygon, nvert, ancho, alto, numPoints, h_points);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float t_calculateArea;
    cudaEventElapsedTime(&t_calculateArea, start, stop);
    printf("Tiempo de ejecucion del kernel: %.3f ms (%d puntos)\n", t_calculateArea, numPoints);

    printf("El area del poligono es: %f\n", area);
    double areaReal =  calculateRealArea(polygon, nvert); 
    printf("El area real es: %f\n", areaReal);
    double err = (fabs(areaReal - area) / area) * 100;
    printf("Error relativo: %f\n", err);

    Punto *d_polygon;
    Punto *d_points;

    cudaMalloc((void**)&d_polygon, nvert * sizeof(Punto));
    cudaMemcpy(d_polygon, polygon, nvert * sizeof(Punto), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_points, numPoints * sizeof(Punto));
    cudaMemcpy(d_points, h_points, numPoints * sizeof(Punto), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock -1) / threadsPerBlock;

    texturaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_texture, WINDOW_WIDTH, WINDOW_HEIGHT, d_polygon, nvert, d_points, numPoints);

    unsigned char *h_texture = (unsigned char *)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(unsigned char));
    int quit = 0;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }

        cudaMemcpy(h_texture, d_texture, WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        SDL_UpdateTexture(texture, NULL, h_texture, WINDOW_WIDTH * 4);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    free(h_texture);
    cudaFree(d_texture);
    cudaFree(d_polygon);
    cudaFree(d_points);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}