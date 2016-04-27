__kernel void kRemap(__global uchar4* src, __global uchar4* dst, __global float *map_x, __global float *map_y){
	unsigned int ix = get_global_id(0);
	unsigned int iy = get_global_id(1);
	unsigned int size_x = get_global_size(0);
	unsigned int size_y = get_global_size(1);

	unsigned int n = iy * size_x + ix;

	uint x = (uint)map_x[n];
	uint y = (uint)map_y[n];

	uint m;
	if ((x < size_x) && (y < size_y))
		m = y * size_x + x;
	else
		m = n;

	dst[n] = src[m];
}