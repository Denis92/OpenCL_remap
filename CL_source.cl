__kernel void kRemap(__global uchar* src, __global uchar* dst, __global int* map_x, __global int *map_y){
	unsigned int n = get_global_id(0);

	int x = map_x[n];
	int y = map_y[n];

	int m = x * 626 + y;

	dst[n * 3] = src[m * 3];
	dst[n * 3 + 1] = src[m * 3 + 1];
	dst[n * 3 + 2] = src[m * 3 + 2];
}