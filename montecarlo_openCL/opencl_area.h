#ifndef OPENCL_AREA_H
#define OPENCL_AREA_H

#include <CL/cl.h>
#include "common.h"

int ejecutarCalculoArea(Punto *polygon, int nvert, Punto *points, int *results, int numPoints);

#endif
