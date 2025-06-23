#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "cl.h"
#include "cl_platform.h"
#define DATAFILE        "p6.data"

#ifndef DATASIZE
#define DATASIZE        4*1024*1024
#endif

#ifndef LOCALSIZE
#define LOCALSIZE       8
#endif

#define NUMGROUPS       DATASIZE/LOCALSIZE

cl_platform_id Platform;
cl_device_id Device;
cl_kernel Kernel;
cl_program Program;
cl_context Context;
cl_command_queue CmdQueue;

float hX[DATASIZE], hY[DATASIZE];
float hSumx4[DATASIZE], hSumx3[DATASIZE], hSumx2[DATASIZE], hSumx[DATASIZE];
float hSumxy[DATASIZE], hSumx2y[DATASIZE], hSumy[DATASIZE];

const char* CL_FILE_NAME = "proj06.cl";

void SelectOpenclDevice();
void Wait(cl_command_queue);
float Determinant(float[3], float[3], float[3]);
void Solve(float[3][3], float[3], float[3]);
void Solve3(float, float, float, float, float, float, float, int, float*, float*, float*);

int main(int argc, char* argv[]) {
    FILE* fp = fopen(CL_FILE_NAME, "r");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
        return 1;
    }

    cl_int status;
    SelectOpenclDevice();

    FILE* fdata = fopen(DATAFILE, "r");
    if (fdata == NULL) {
        fprintf(stderr, "Cannot open data file '%s'\n", DATAFILE);
        return -1;
    }

    for (int i = 0; i < DATASIZE; i++) {
        fscanf(fdata, "%f %f", &hX[i], &hY[i]);
    }
    fclose(fdata);

    Context = clCreateContext(NULL, 1, &Device, NULL, NULL, &status);
    CmdQueue = clCreateCommandQueue(Context, Device, 0, &status);

    size_t xySize = DATASIZE * sizeof(float);

    cl_mem dX = clCreateBuffer(Context, CL_MEM_READ_ONLY, xySize, NULL, &status);
    cl_mem dY = clCreateBuffer(Context, CL_MEM_READ_ONLY, xySize, NULL, &status);
    cl_mem dSumx4 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx3 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx2 = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumx2y = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumxy = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);
    cl_mem dSumy = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, xySize, NULL, &status);

    clEnqueueWriteBuffer(CmdQueue, dX, CL_FALSE, 0, xySize, hX, 0, NULL, NULL);
    clEnqueueWriteBuffer(CmdQueue, dY, CL_FALSE, 0, xySize, hY, 0, NULL, NULL);
    Wait(CmdQueue);

    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* clProgramText = new char[fileSize + 1];
    fread(clProgramText, 1, fileSize, fp);
    clProgramText[fileSize] = '\0';
    fclose(fp);

    const char* strings[] = { clProgramText };
    Program = clCreateProgramWithSource(Context, 1, strings, NULL, &status);
    delete[] clProgramText;

    status = clBuildProgram(Program, 1, &Device, "", NULL, NULL);
    Kernel = clCreateKernel(Program, "Regression", &status);

    clSetKernelArg(Kernel, 0, sizeof(cl_mem), &dX);
    clSetKernelArg(Kernel, 1, sizeof(cl_mem), &dY);
    clSetKernelArg(Kernel, 2, sizeof(cl_mem), &dSumx4);
    clSetKernelArg(Kernel, 3, sizeof(cl_mem), &dSumx3);
    clSetKernelArg(Kernel, 4, sizeof(cl_mem), &dSumx2);
    clSetKernelArg(Kernel, 5, sizeof(cl_mem), &dSumx);
    clSetKernelArg(Kernel, 6, sizeof(cl_mem), &dSumx2y);
    clSetKernelArg(Kernel, 7, sizeof(cl_mem), &dSumxy);
    clSetKernelArg(Kernel, 8, sizeof(cl_mem), &dSumy);

    size_t globalWorkSize[3] = { DATASIZE, 1, 1 };
    size_t localWorkSize[3] = { LOCALSIZE, 1, 1 };

    Wait(CmdQueue);
    double time0 = omp_get_wtime();

    clEnqueueNDRangeKernel(CmdQueue, Kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    Wait(CmdQueue);

    double time1 = omp_get_wtime();

    clEnqueueReadBuffer(CmdQueue, dSumx, CL_FALSE, 0, xySize, hSumx, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumx2, CL_FALSE, 0, xySize, hSumx2, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumx3, CL_FALSE, 0, xySize, hSumx3, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumx4, CL_FALSE, 0, xySize, hSumx4, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumy, CL_FALSE, 0, xySize, hSumy, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumxy, CL_FALSE, 0, xySize, hSumxy, 0, NULL, NULL);
    clEnqueueReadBuffer(CmdQueue, dSumx2y, CL_FALSE, 0, xySize, hSumx2y, 0, NULL, NULL);
    Wait(CmdQueue);

    float sumx = 0., sumx2 = 0., sumx3 = 0., sumx4 = 0.;
    float sumxy = 0., sumx2y = 0., sumy = 0.;
    for (int i = 0; i < DATASIZE; i++) {
        sumx += hSumx[i];
        sumx2 += hSumx2[i];
        sumx3 += hSumx3[i];
        sumx4 += hSumx4[i];
        sumy += hSumy[i];
        sumxy += hSumxy[i];
        sumx2y += hSumx2y[i];
    }

    float Q, L, C;
    Solve3(sumx4, sumx3, sumx2, sumx, sumx2y, sumxy, sumy, DATASIZE, &Q, &L, &C);
    fprintf(stderr, "Array Size: %8d , Work Elements: %4d , MegaPointsProcessedPerSecond: %10.2lf, (%7.1f,%7.1f,%7.1f)\n",
        DATASIZE, LOCALSIZE, (double)DATASIZE / (time1 - time0) / 1000000., Q, L, C);

    clReleaseKernel(Kernel);
    clReleaseProgram(Program);
    clReleaseCommandQueue(CmdQueue);
    clReleaseMemObject(dX);
    clReleaseMemObject(dY);
    clReleaseMemObject(dSumx4);
    clReleaseMemObject(dSumx3);
    clReleaseMemObject(dSumx2);
    clReleaseMemObject(dSumx);
    clReleaseMemObject(dSumx2y);
    clReleaseMemObject(dSumxy);
    clReleaseMemObject(dSumy);

    return 0;
}

// Supporting Functions

void Wait(cl_command_queue queue) {
    cl_event wait;
    clEnqueueMarker(queue, &wait);
    clWaitForEvents(1, &wait);
}

#define ID_AMD 0x1002
#define ID_INTEL 0x8086
#define ID_NVIDIA 0x10de

void SelectOpenclDevice() {
    int bestPlatform = -1, bestDevice = -1;
    cl_device_type bestDeviceType;
    cl_uint bestDeviceVendor;
    cl_int status;
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    for (int p = 0; p < (int)numPlatforms; p++) {
        cl_uint numDevices;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        cl_device_id* devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        for (int d = 0; d < (int)numDevices; d++) {
            cl_device_type type;
            cl_uint vendor;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);

            if (bestPlatform < 0 || (bestDeviceType == CL_DEVICE_TYPE_CPU && type == CL_DEVICE_TYPE_GPU)
                || (bestDeviceVendor == ID_INTEL && (vendor == ID_NVIDIA || vendor == ID_AMD))) {
                bestPlatform = p;
                bestDevice = d;
                Platform = platforms[p];
                Device = devices[d];
                bestDeviceType = type;
                bestDeviceVendor = vendor;
            }
        }
        delete[] devices;
    }
    delete[] platforms;
}

float Determinant(float c0[3], float c1[3], float c2[3]) {
    return c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
        - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
}

void Solve(float A[3][3], float X[3], float B[3]) {
    float col0[3] = { A[0][0], A[1][0], A[2][0] };
    float col1[3] = { A[0][1], A[1][1], A[2][1] };
    float col2[3] = { A[0][2], A[1][2], A[2][2] };
    float d0 = Determinant(col0, col1, col2);
    float dq = Determinant(B, col1, col2);
    float dl = Determinant(col0, B, col2);
    float dc = Determinant(col0, col1, B);
    X[0] = dq / d0;
    X[1] = dl / d0;
    X[2] = dc / d0;
}

void Solve3(float sumx4, float sumx3, float sumx2, float sumx, float sumx2y, float sumxy, float sumy, int datasize, float* Q, float* L, float* C) {
    float A[3][3] = {
        {sumx4, sumx3, sumx2},
        {sumx3, sumx2, sumx},
        {sumx2, sumx, (float)datasize}
    };
    float Y[3] = { sumx2y, sumxy, sumy }, X[3];
    Solve(A, X, Y);
    *Q = X[0]; *L = X[1]; *C = X[2];
}