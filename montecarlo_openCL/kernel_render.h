#ifndef KERNEL_RENDER_H
#define KERNEL_RENDER_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ejecutarTextura(Punto *polygon, int nvert, int numPoints, int ancho, int alto);

#ifdef __cplusplus
}
#endif

#endif