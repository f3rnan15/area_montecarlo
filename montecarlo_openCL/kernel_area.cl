typedef struct{
    int x, y;
} Punto;

__kernel void calculateArea(__global const Punto *polygon, const int nvert, __global const Punto *points,
                            __global int *results, const int numPoints){
    int idx = get_global_id(0);
    if (idx >= numPoints) return;

    Punto punto = points[idx];
    int winding_number = 0;

    for (int i = 0; i < nvert; i++){
        Punto p1 = polygon[i];
        Punto p2 = polygon[(i + 1) % nvert];

        if (p1.y <= punto.y) {
            if (p2.y > punto.y && ((p2.x - p1.x) * (punto.y - p1.y) - (punto.x - p1.x) * (p2.y - p1.y)) > 0)
                winding_number++;
        } else {
            if (p2.y <= punto.y && ((p2.x - p1.x) * (punto.y - p1.y) - (punto.x - p1.x) * (p2.y - p1.y)) < 0)
                winding_number--;
        }
    }

    results[idx] = (winding_number != 0);
}
