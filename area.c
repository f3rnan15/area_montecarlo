#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <omp.h>
#include <math.h>

typedef struct {
    int x, y;
} Punto;

Punto calcularCentroide(Punto puntos[], int n) {
    Punto centroide = {0, 0};
    for (int i = 0; i < n; i++) {
        centroide.x += puntos[i].x;
        centroide.y += puntos[i].y;
    }
    centroide.x /= n;
    centroide.y /= n;
    return centroide;
}

double calcularAngulo(Punto p, Punto centroide) {
    return atan2(p.y - centroide.y, p.x - centroide.x);
}

int compararPuntos(const void *a, const void *b) {
    Punto *p1 = (Punto *)a;
    Punto *p2 = (Punto *)b;
    Punto centroide = calcularCentroide((Punto *)a, 2);
    double angulo1 = calcularAngulo(*p1, centroide);
    double angulo2 = calcularAngulo(*p2, centroide);
    return (angulo1 < angulo2) ? -1 : (angulo1 > angulo2) ? 1 : 0;
}

Punto* obtenerPoligono(Punto puntos[], int n) {
    Punto *puntosOrdenados = malloc(n * sizeof(Punto));
    if (puntosOrdenados == NULL) {
        fprintf(stderr, "Error al asignar memoria\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        puntosOrdenados[i] = puntos[i];
    }

    Punto centroide = calcularCentroide(puntos, n);

    qsort(puntosOrdenados, n, sizeof(Punto), compararPuntos);

    return puntosOrdenados;
}

int inicializarSDL(SDL_Window **window, SDL_Renderer **renderer, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Error al inicializar SDL: %s\n", SDL_GetError());
        return -1;
    }

    *window = SDL_CreateWindow("Envolvente Convexa", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
    if (*window == NULL) {
        printf("Error al crear la ventana: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);
    if (*renderer == NULL) {
        SDL_DestroyWindow(*window);
        printf("Error al crear el renderizador: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    return 0;
}

void dibujar(SDL_Renderer *renderer, Punto puntos[], int n, Punto stack[], int stack_size, Punto *dentro, int conDentro, Punto *fuera, int conFuera) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    for (int i = 0; i < n; i++) {
        SDL_RenderDrawPoint(renderer, puntos[i].x, puntos[i].y);
    }

    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    for (int i = 0; i < stack_size; i++) {
        SDL_RenderDrawLine(renderer, stack[i].x, stack[i].y, stack[(i + 1) % stack_size].x, stack[(i + 1) % stack_size].y);
    }

    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    for (int i = 0; i < conDentro; i++) {
        SDL_RenderDrawPoint(renderer, (int)dentro[i].x, (int)dentro[i].y);
    }

    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
    for (int i = 0; i < conFuera; i++) {
        SDL_RenderDrawPoint(renderer, (int)fuera[i].x, (int)fuera[i].y);
    }

    SDL_RenderPresent(renderer);
}

int orientacion2(Punto p, Punto q, Punto r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : -1;
}

int puntoEnPoligono(Punto polygon[], int n, Punto point) {
    int winding_number = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Punto p1 = polygon[i];
        Punto p2 = polygon[(i + 1) % n];

        if (p1.y <= point.y) {
            if (p2.y > point.y && orientacion2(p1, p2, point) == -1) {
                #pragma omp atomic
                winding_number++;
            }
        } else {
            if (p2.y <= point.y && orientacion2(p1, p2, point) == 1) {
                #pragma omp atomic
                winding_number--;
            }
        }
    }

    return winding_number != 0;
}

double calculateArea(int numPoints, int nvert, Punto *vert, int ancho, int alto, Punto *dentro, Punto *fuera, int *contDentro) {
    int *localCounts = (int *)calloc(omp_get_max_threads(), sizeof(int));
    if (localCounts == NULL) {
        printf("Error al asignar memoria para localCounts\n");
        return -1;
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int seed = thread_id;
        int localCountInside = 0;

        #pragma omp for
        for (int i = 0; i < numPoints; i++) {
            double x = (double)rand() / RAND_MAX * ancho;
            double y = (double)rand() / RAND_MAX * alto;
            Punto punto = {x, y};

            if (puntoEnPoligono(vert, nvert, punto)) {
                int index = thread_id * (numPoints / omp_get_num_threads()) + localCountInside++;
                dentro[index] = punto;
            } else {
                int index = i - localCountInside;
                fuera[index] = punto;
            }
        }
        localCounts[thread_id] = localCountInside;
    }

    int countInside = 0;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        countInside += localCounts[i];
    }
    free(localCounts);

    *contDentro = countInside;
    double area = (double)countInside / numPoints * ancho * alto;
    return area;
}

int main() {
    Punto puntos[] = {{100, 100}, {200, 200}, {100, 200}, {200, 100}};
    int n = sizeof(puntos) / sizeof(puntos[0]);

    int alto = 400;
    int ancho = 400;

    Punto *stack = obtenerPoligono(puntos, n);

    printf("Puntos ordenados:\n");
    for (int i = 0; i < n; i++) {
        printf("(%d, %d)\n", stack[i].x, stack[i].y);
    }

    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;

    if (inicializarSDL(&window, &renderer, ancho, alto) != 0) {
        return -1;
    }

    long int numPoints = 10000000;

    Punto *dentro = (Punto *)malloc(numPoints * sizeof(Punto));
    if (dentro == NULL) {
        printf("No se pudo asignar memoria para los puntos aleatorios-1.\n");
        return -1;
    }
    Punto *fuera = (Punto *)malloc(numPoints * sizeof(Punto));
    if (fuera == NULL) {
        printf("No se pudo asignar memoria para los puntos aleatorios-2.\n");
        return -1;
    }
    int conDentro;
    double area = calculateArea(numPoints, n, stack, ancho, alto, dentro, fuera, &conDentro);
    printf("El area del poligono es: %f\n", area);

    int conFuera = numPoints - conDentro;
    int quit = 0;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }
        dibujar(renderer, puntos, n, stack, n, dentro, conDentro, fuera, conFuera);
    }

    free(dentro);
    free(fuera);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
