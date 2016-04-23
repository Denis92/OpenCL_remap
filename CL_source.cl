__kernel void kRemap(__global uchar4* src, __global uchar4* dst, __global int* map_x, __global int *map_y, __global int *width){
	unsigned int n = get_global_id(0);

	int x = map_x[n];
	int y = map_y[n];

	int m = x * *width + y;

	dst[n] = src[m];
	dst[n].s3 = 0;
}