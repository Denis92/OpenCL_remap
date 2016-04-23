
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <CL/cl.h>

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
		resSize += sBuf.size() + 1;
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


void remapCL(Mat *src, Mat *dst, Mat *map_x, Mat *map_y){


	Mat src_proc, dst_proc;
	cvtColor(*src, src_proc, CV_RGB2RGBA);
	dst_proc.create(src_proc.size(), src_proc.type());

	const char *fn = "CL_source.cl";
	const char *source;
	source = load_cl_source(fn);
	if (!source) return;
	//printf("%s\n", source);

	int err;                            // error code returned from api calls

	//Get an OpenCL platform
	cl_platform_id cpPlatform;
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if (err != CL_SUCCESS){

		printf("Error: Failed to create a device group!\n");
		return;
	}

	// Get a GPU device
	cl_device_id cdDevice;
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	if (err != CL_SUCCESS){

		printf("Error: Failed to create a compute context!\n");
		return;
	}

	cl_context GPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &err);
	cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);

	cl_mem GPUSrcVector = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar4) * src_proc.rows * src_proc.cols, src_proc.data, NULL);
	cl_mem GPUMapXVector = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * dst_proc.rows * dst_proc.cols, map_x->data, NULL);
	cl_mem GPUMapYVector = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * dst_proc.rows * dst_proc.cols, map_y->data, NULL);
	cl_mem GPUDstVector = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_uchar4) * dst_proc.rows * dst_proc.cols, NULL, NULL);
	cl_mem GPUWidth = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &dst_proc.cols, NULL);

	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 1, &source, NULL, NULL);
	clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	cl_kernel OpenCLRemap = clCreateKernel(OpenCLProgram, "kRemap", NULL);

	clSetKernelArg(OpenCLRemap, 0, sizeof(cl_mem), (void*)&GPUSrcVector);
	clSetKernelArg(OpenCLRemap, 1, sizeof(cl_mem), (void*)&GPUDstVector);
	clSetKernelArg(OpenCLRemap, 2, sizeof(cl_mem), (void*)&GPUMapXVector);
	clSetKernelArg(OpenCLRemap, 3, sizeof(cl_mem), (void*)&GPUMapYVector);
	clSetKernelArg(OpenCLRemap, 4, sizeof(cl_mem), (void*)&GPUWidth);

	size_t WorkSize[1] = { src->rows * src->cols };
	clEnqueueNDRangeKernel(cqCommandQueue, OpenCLRemap, 1, NULL, WorkSize, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(cqCommandQueue, GPUDstVector, CL_TRUE, 0, sizeof(cl_uchar4) * dst_proc.rows * dst_proc.cols, dst_proc.data, 0, NULL, NULL);

	clReleaseKernel(OpenCLRemap);
	clReleaseProgram(OpenCLProgram);
	clReleaseCommandQueue(cqCommandQueue);
	clReleaseContext(GPUContext);
	clReleaseMemObject(GPUSrcVector);
	clReleaseMemObject(GPUDstVector);
	clReleaseMemObject(GPUMapXVector);
	clReleaseMemObject(GPUMapYVector);

	delete[] source;

	cvtColor(dst_proc, *dst, CV_RGBA2RGB);
}

void remapCPU(Mat *src, Mat *dst, Mat *map_x, Mat *map_y){
	for (int j = 0; j < src->rows; j++)
	{
		for (int i = 0; i < src->cols; i++)
		{
			int src_x = map_x->at<int>(j, i);
			int src_y = map_y->at<int>(j, i);
			if (src_x >= src->rows || src_y >= src->cols) continue;
			dst->at<Vec3b>(j, i) = src->at<Vec3b>(src_y, src_x);
		}
	}
}

int main(int argc, char* argv[]){

	Mat src, dst;
	Mat map_x, map_y;

	src = imread("..\\src.jpg", 1);
	dst.create(src.size(), src.type());
	map_x.create(src.size(), CV_32FC1);
	map_y.create(src.size(), CV_32FC1);

	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{

			dst.at<Vec3b>(j, i) = Vec3b(200, 200, 10);

			if (i % 2 == 0)
				map_x.at<int>(j, i) = i;
			else
				map_x.at<int>(j, i) = src.cols - i;

			if (j % 2 == 0)
				map_y.at<int>(j, i) = j;
			else{
				map_y.at<int>(j, i) = src.rows - j;
			}

			//map_x.at<int>(j, i) = src.cols - i - 1;
			//map_y.at<int>(j, i) = src.rows - j - 1;
		}
	}

	remapCL(&src, &dst, &map_x, &map_y);

	imwrite("..\\dst.jpg", dst);


	//--------------------------------------------------------------
	system("PAUSE");
	return 0;
}