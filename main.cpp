#include <iostream>
#include <CL/cl.h>

#define SIZE 10

const char* OpenCLSource[] = {
	"__kernel void kRemap(__global int* src, __global int* dst, __global int* map)",
	"{",
	" unsigned int n = get_global_id(0);",
	" dst[n] = src[map[n]];",
	"}"
};

int main(int argc, char* argv[]){

	int srcVec[SIZE], dstVec[SIZE], mapVec[SIZE];

	for (int i = 0; i < SIZE; i++){
		srcVec[i] = i;
		dstVec[i] = 0;
		mapVec[i] = i%3;
	}

	int err;                            // error code returned from api calls

	//Get an OpenCL platform
	cl_platform_id cpPlatform;
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if (err != CL_SUCCESS){

		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Get a GPU device
	cl_device_id cdDevice;
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	if (err != CL_SUCCESS){

		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	cl_context GPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &err);
	cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);

	cl_mem GPUSrcVector = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, srcVec, NULL);
	cl_mem GPUMapVector = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, mapVec, NULL);
	cl_mem GPUDstVector = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY,	sizeof(int) * SIZE, NULL, NULL);

	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 5, OpenCLSource, NULL, NULL);
	clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	cl_kernel OpenCLRemap = clCreateKernel(OpenCLProgram, "kRemap", NULL);

	clSetKernelArg(OpenCLRemap, 0, sizeof(cl_mem), (void*)&GPUSrcVector);
	clSetKernelArg(OpenCLRemap, 1, sizeof(cl_mem), (void*)&GPUDstVector);
	clSetKernelArg(OpenCLRemap, 2, sizeof(cl_mem), (void*)&GPUMapVector);

	size_t WorkSize[1] = { SIZE };
	clEnqueueNDRangeKernel(cqCommandQueue, OpenCLRemap, 1, NULL, WorkSize, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(cqCommandQueue, GPUDstVector, CL_TRUE, 0, SIZE * sizeof(int), dstVec, 0, NULL, NULL);

	clReleaseKernel(OpenCLRemap);
	clReleaseProgram(OpenCLProgram);
	clReleaseCommandQueue(cqCommandQueue);
	clReleaseContext(GPUContext);
	clReleaseMemObject(GPUSrcVector);
	clReleaseMemObject(GPUDstVector);
	clReleaseMemObject(GPUMapVector);

	for (int i = 0; i < SIZE; i++)
		printf("Src: %d, Dst:%d\n", srcVec[i], dstVec[i]);

	//--------------------------------------------------------------
	system("PAUSE");
	return 0;
}