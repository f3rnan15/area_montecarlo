#include <CL/cl.h>
#include <stdio.h>
#include <time.h>
#include "opencl_area.h"

int ejecutarCalculoArea(Punto *polygon, int nvert, Punto *points, int *results, int numPoints){
    clock_t start_func, end_func;
    double ms_func;

    start_func = clock();
    
    FILE *f = fopen("kernel_area.cl", "r");
    if (!f){
        printf("Error al abrir kernel_area.cl\n");
        return -1;
    }

    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    rewind(f);
    char *kernel_src = (char *)malloc(kernel_size + 1);
    if (!kernel_src){
        printf("Error al asignar memoria para el kernel(opencl)\n");
        fclose(f);
        return -1;
    }
    size_t read_elements = fread(kernel_src, 1, kernel_size, f);
    if (read_elements != kernel_size){
        printf("Error al leer el kernel del archivo(opencl)\n");
        fclose(f);
        free(kernel_src);
        return -1;
    }
    kernel_src[kernel_size] = '\0';
    fclose(f);

    // inicializa opencl
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS){
        printf("Error al obtener la platafroma\n");
        return -1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS){
        printf("Error al obtener el dispositivo\n");
        return -1;
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS){
        printf("Error al crear el contexto\n");
        return -1;
    }

    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS){
        printf("Error al crear la cola de comandos\n");
        clReleaseContext(context);
        return -1;
    }

    // crea y compila el programa
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, &kernel_size, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculateArea", NULL);

    // buffers
    cl_mem polygon_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nvert * sizeof(Punto), polygon, NULL);
    cl_mem points_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numPoints * sizeof(Punto), points, NULL);
    cl_mem results_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numPoints * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, polygon_buffer, CL_TRUE, 0, nvert * sizeof(Punto), polygon, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, points_buffer, CL_TRUE, 0, numPoints * sizeof(Punto), points, 0, NULL, NULL);

    // argumentos del kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &polygon_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &nvert);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &points_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &results_buffer);
    clSetKernelArg(kernel, 4, sizeof(int), &numPoints);

    clock_t start_kernel, end_kernel;
    double ms_kernel;

    start_kernel = clock();

    // ejecuta el kernel
    size_t size = numPoints;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);

    // lee resultado
    clEnqueueReadBuffer(queue, results_buffer, CL_TRUE, 0, numPoints * sizeof(int), results, 0, NULL, NULL);

    /*
    for (int i = 0; i < numPoints; i++)
        printf(", results[%d]=%d", i, results[i]);
    */
    
    end_kernel = clock();
    ms_kernel = (double)(end_kernel - start_kernel) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución del kernel calculateArea: %.3f ms\n", ms_kernel * 1000);

    int countInside = 0;
    for (int i = 0; i < numPoints; i++){
        if (results[i] == 1) countInside++;
    }

    // libera recursos
    clReleaseMemObject(polygon_buffer);
    clReleaseMemObject(points_buffer);
    clReleaseMemObject(results_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_src);

    end_func = clock();
    ms_func = (double)(end_func - start_func) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución de ejecutarCalculateArea: %.3f ms\n", ms_func * 1000);

    return countInside;
}