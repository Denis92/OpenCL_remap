__kernel void kRemap(__global int* src, __global int* dst, __global int* map){
	unsigned int n = get_global_id(0);
	dst[n] = src[map[n]];
}