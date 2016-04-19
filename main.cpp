
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include <iomanip>
//#include <stdlib.h>

#include <CL/cl.h>

#define u_char unsigned char

#define SIZE 10

using namespace cv;
using namespace std;

char * load_cl_source(const char *filename){
	// !!! INCLUDE <string> <vector>
	fstream f;
	f.open(filename);
	if (!f) return NULL;
	string sBuf;

	vector<string> vres;
	vres.clear();

	int resSize = 0;
	while (!f.eof()){
		f >> sBuf;
		vres.push_back(sBuf);
		vres.push_back(" ");
		resSize += sBuf.size() + 1; // DEBUG
	}
	
	if (!resSize) return NULL;

	char *res = new char[resSize];
	strcpy_s(res, 1, "");
	resSize = 1;
	for (int i = 0; i < vres.size() - 1; i++){
		resSize += vres[i].length();
		strcat_s(res, resSize, vres[i].c_str());
	}
	f.close();
	return res;
};

/*
const char* OpenCLSource[] = {
	"__kernel void kRemap(__global int* src, __global int* dst, __global int* map)",
	"{",
	" unsigned int n = get_global_id(0);",
	" dst[n] = src[map[n]];",
	"}"
};*/

int main(int argc, char* argv[]){

	Mat src = imread("D:\\Workspace\\cpp\\opencv_test\\tr.jpg", 1);
	for (int i = 0; i < 1200; i++){
		u_char buf = src.data[i];
		//printf("%i", (int)buf);
	}
	printf("%i", src.rows);
	printf("\n");

	const char *fn = "CL_source.cl";
	const char *source;
	source = load_cl_source(fn);
	if (!source) return EXIT_FAILURE;
	//printf("%s\n", source);

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

	//cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 5, OpenCLSource, NULL, NULL);
	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 1, &source, NULL, NULL);
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

	delete[] source;

	//char *windowName = "Remap";
	//namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	//imshow(windowName, src);

	for (int i = 0; i < SIZE; i++)
		printf("Src: %d, Dst:%d\n", srcVec[i], dstVec[i]);

	//--------------------------------------------------------------
	system("PAUSE");
	return 0;
}