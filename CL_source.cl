__kernel void kRemap(__global uchar4* src, __global uchar4* dst, __global int* map_x, __global int *map_y){
	unsigned int ix = get_global_id(0);
	unsigned int iy = get_global_id(1);
	unsigned int size_x = get_global_size(0);

	unsigned int n = ix * size_x + iy;

	int x = map_x[n];
	int y = map_y[n];

	int m = x * size_x + y;

	dst[n] = src[m];

}