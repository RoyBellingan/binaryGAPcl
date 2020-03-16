#include <bitset>
#include <chrono>
#include <iostream>

//using namespace std;

//int main() {
//	uint64_t v;
//	//	std::cin >> v;

//	auto start = chrono::steady_clock::now();

//	uint64_t longest = 0;
//	uint64_t longest4num = 0;
//	for (int var = 1; var < 2E7; ++var) {
//		uint64_t max = 0;
//		uint64_t cur = 0;
//		//used to mark after we find the first set bit, to skip the initial 0 sequence
//		bool primed = false;
//		//to print at the end only from setted MSB
//		//std::string str;
//		//start scanning from MSB (most significant bit)
//		auto msb = (sizeof(v) * 8) - __builtin_clzl(var);
//		for (int i = msb - 1; i >= 0; i--) {
//			uint64_t mask = 1ull << i;
//			if (!(var & mask)) {
//				if (primed) {
//					//str.append("0");
//					cur++;
//				}
//			} else {
//				max = std::max(max, cur);
//				//str.append("1");
//				primed = true;
//				cur    = 0;
//			}
//		}

//		if(max > longest){
//			longest = max;
//			longest4num = var;
//		}
//	}

//	auto end = chrono::steady_clock::now();

//	auto ns =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();
//	printf("Elapsed time : %ld ns, %.3f s \n",ns,ns/1E9);
//	cout << "longest sequence: " << longest << " for " << longest4num << endl;
//	//cout << "max sequence lenght in " << v << " is " << max << " in " << str << endl;
//	return 0;
//}

#define PROGRAM_FILE "bitGap.cl"
#define KERNEL_FUNC "computeGap"
#define ARRAY_SIZE 20480

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
	cl_uint         platformCount;
	cl_platform_id* platforms;
	cl_uint         deviceCount;
	cl_device_id    dev;
	cl_device_id*   devices;
	int             err;

	// get platform count
	clGetPlatformIDs(NULL, NULL, &platformCount);

	// get all platforms
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	err       = clGetPlatformIDs(platformCount, platforms, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	// get all devices
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	err     = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	if (err < 0) {
		perror("Couldn't access any devices");
	}

	// print device name
	size_t valueSize = 0;
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
	char* value = (char*)malloc(valueSize);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	return devices[0];
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE*      program_handle;
	char *     program_buffer, *program_log;
	size_t     program_size, log_size;
	int        err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer               = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
										(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
							  0, NULL, &log_size);
		program_log           = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
							  log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

int main() {
	struct timespec res1, res2, res3;
	//clock_getres(CLOCK_PROCESS_CPUTIME_ID, &res);

	/* OpenCL structures */
	cl_device_id     device;
	cl_context       context;
	cl_program       program;
	cl_kernel        kernel;
	cl_command_queue queue;
	cl_int           err;

	/* Data and buffers */
	//localsize should be a multiple of 32, try to keep the block size of data lower than 32K element to havoid big buffer swap
	//we aim to compute 2E32 - 1

	//let's try 65K job
	auto   kernelLoop  = 1 << 20;
	size_t global_size = (std::numeric_limits<uint32_t>::max() / kernelLoop) + 1; //pedantic matematical stuff, +1 because we count from 0
	//size_t global_size = 64;
	size_t local_size = 32;
	//	size_t global_size = 2;
	//	size_t local_size  = 1;
	auto bufSize = (1 + global_size) * 4;
	//allocate buffer
	cl_mem numberBuf, gapBuf;

	/* Create device and context */
	device  = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	program = build_program(context, device, PROGRAM_FILE);

	uint* numBuf2 = (uint*)calloc(bufSize, 1);
	uint* gapBuf2 = (uint*)calloc(bufSize, 1);
	numBuf2[0]    = 99;
	gapBuf2[0]    = 97;

	//	/* Create data buffer */
	numberBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufSize, numBuf2, &err);
	gapBuf    = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bufSize, gapBuf2, &err);

	numBuf2[0] = 1;
	gapBuf2[0] = 1;

	//	//this is sligtly larger to accomodate a scratch are to circumvent a bug in Intel card that disable atomic function
	//	//for var declare IN the kernel
	//	nuclideBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
	//					   CL_MEM_COPY_HOST_PTR, 8 * sizeof(uint32_t), nuclides, &err);
	//	probBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
	//					CL_MEM_COPY_HOST_PTR, 2 * sizeof(float), prob, &err);
	//	seedBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
	//					CL_MEM_COPY_HOST_PTR, sizeof(uint32_t), &seed, &err);

	if (err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	};

	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if (err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	};

	//	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &numberBuf);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gapBuf);
	//	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &probBuffer);
	//	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &seedBuffer);

	if (err < 0) {
		perror("Couldn't create a kernel argument");
		exit(1);
	}
	//printf("Radium, delta, Radon, delta, elapsed \n");

	/* Enqueue kernel */

	auto start = std::chrono::steady_clock::now();

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
								 &local_size, 0, NULL, NULL);
	auto finish = std::chrono::steady_clock::now();


	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}

	//		auto old1 = nuclides[0];
	//		int  old2 = nuclides[1];

	/* Read the kernel's output */
	err = clEnqueueReadBuffer(queue, numberBuf, CL_TRUE, 0,
							  bufSize, numBuf2, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, gapBuf, CL_TRUE, 0,
							  bufSize, gapBuf2, 0, NULL, NULL);

	auto ns =  std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();

	if (err < 0) {
		perror("Couldn't read the buffer");
		exit(1);
	}


	auto bigPtr = std::max_element(gapBuf2, gapBuf2 + global_size);
	auto bigPos = bigPtr - gapBuf2;
	std::cout << "The biggest gap is " << *bigPtr << " at post " << bigPos << " for " << numBuf2[bigPos] << "\n";
	printf("Elapsed time : %ld ns, %.6f s \n",ns,ns/1E9);

//	for (int var = 0; var < global_size; ++var) {
//		std::cout << "block " << numBuf2[var] << " max " << gapBuf2[var] <<"\n" ;
//	}
	/* Deallocate resources */
	clReleaseKernel(kernel);
	//	clReleaseMemObject(probBuffer);
	//	clReleaseMemObject(iterBuffer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
