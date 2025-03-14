#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "opencl_area.h"
#include "kernel_render.h"
#include "common.h"

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

double calculateRealArea(Punto *polygon, int nvert) {
    double area = 0;

    for (int i = 0; i < nvert; i++) {
        int j = (i + 1) % nvert;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[i].y * polygon[j].x;
    }

    return fabs(area) / 2.0;
}

int main() {
    Punto polygon[] = {{100, 100}, {200, 200}, {100, 200}, {200, 100}};
    //Punto polygon[] = {{50, 50},{150, 75},{125, 150},{200, 200},{100, 250},{50, 200},{25, 100}};
    int nvert = sizeof(polygon) / sizeof(Punto);

    int ancho = 400;
    int alto = 400;
    int numPoints = 100000000;

    printf("Numero de puntos: %d\n", numPoints);

    ordenarPuntos(polygon, nvert);

    Punto *points = (Punto*)malloc(numPoints * sizeof(Punto));
    if (!points){
        printf("Error al reservar memoria para los puntos\n");
        return -1;
    }

    for (int i = 0; i < numPoints; i++){
        double x = (double)rand() / RAND_MAX * ancho;
        double y = (double)rand() / RAND_MAX * alto;
        points[i].x = (int)x;
        points[i].y = (int)y;
    }

    int *results = malloc(numPoints * sizeof(int));
    if (!results){
        printf("Error al reservar memoria para los resultados\n");
        free(points);
        return -1;
    }

    int countInside = ejecutarCalculoArea(polygon, nvert, points, results, numPoints);
    double area = (double)countInside / numPoints * ancho * alto;
    printf("Area estimada del poligono %f\n", area);
    double area_real = calculateRealArea(polygon, nvert);
    printf("Area real del poligono: %f\n", area_real);
    double err = (fabs(area_real - area) / area) * 100;
    printf("Error relativo: %f\n", err);

    ejecutarTextura(polygon, nvert, numPoints, ancho, alto);

    free(points);
    free(results);

    return 0;
}