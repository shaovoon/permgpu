// PermOpenCL.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <MMSystem.h>

#pragma comment(lib, "winmm.lib")

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NUMBER_OF_ELEMENTS 5
#define LOCALGROUPS 1024
#define OFFSET 0
// When MAX_PERM = 0, means find all permutations
#define MAX_PERM 0

// Function Prototypes
long long Fact(long long n);
void check(char* arrDest, long long Max);
void display(char* arrDest, long long Max);
bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete [] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		delete [] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete [] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
	char *a, long long* offset, long long* Max)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		(*Max) * NUMBER_OF_ELEMENTS, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(long long), offset, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(long long), Max, NULL);

	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}

	return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
	for (int i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);

}

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[3] = { 0, 0, 0 };
	cl_int errNum;
	// Create an OpenCL context on first available platform
	context = CreateContext();
	if (context == NULL)
	{
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return 1;
	}

	// Create a command-queue on the first device available
	// on the created context
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL program from HelloWorld.cl kernel source
	char file[] = "C:\\Users\\wong\\Documents\\Visual Studio 2010\\Projects\\PermOpenCL\\PermOpenCL\\PermOpenCL.cl";
	program = CreateProgram(context, device, file);
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL kernel
	kernel = clCreateKernel(program, "Permute", NULL);
	if (kernel == NULL)
	{
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	size_t param_value=0;
	size_t param_value_size_ret=0;
	clGetKernelWorkGroupInfo (kernel, device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t),
		(void*)&param_value,
		&param_value_size_ret);

	if(param_value_size_ret!=sizeof(size_t))
	{
		std::cerr << "clGetKernelWorkGroupInfo return different size for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	size_t param_value2=0;
	size_t param_value_size_ret2=0;
	clGetKernelWorkGroupInfo (kernel, device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t),
		(void*)&param_value2,
		&param_value_size_ret2);

	if(param_value_size_ret2!=sizeof(size_t))
	{
		std::cerr << "clGetKernelWorkGroupInfo return different size for CL_KERNEL_WORK_GROUP_SIZE" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	long long offset = OFFSET;
	long long Max = 0;
	
	if(MAX_PERM==0)
		Max = Fact(NUMBER_OF_ELEMENTS);
	else
		Max = MAX_PERM;

	char* arrDest = new char[Max*NUMBER_OF_ELEMENTS];

	if (!CreateMemObjects(context, memObjects, arrDest, &offset, &Max))
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Set the kernel arguments (result, a, b)
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error setting kernel arguments." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	int GlobalGroups = Max/LOCALGROUPS;
	if(Max%LOCALGROUPS != 0)
		++GlobalGroups;

	++GlobalGroups;

	size_t globalWorkSize[1] = { GlobalGroups * LOCALGROUPS };
	int LocalGroups = Max <= LOCALGROUPS ? 1 : LOCALGROUPS;
	size_t localWorkSize[1] = { LocalGroups };

	UINT wTimerRes = 0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime = timeGetTime();

	// Queue the kernel up for execution across the array
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize,
		0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error queuing kernel for execution." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	clFinish(commandQueue);

	DWORD endTime = timeGetTime();
	char buf[50];
	sprintf(buf, "Timing: %dms\n", endTime-startTime);
	std::cout<<buf<<std::endl;

	DestroyMMTimer(wTimerRes, init);

	// Read the output buffer back to the Host
	errNum = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE,
		0, Max*NUMBER_OF_ELEMENTS, arrDest,
		0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	check(arrDest, Max);
	display(arrDest, Max);
	std::cout << std::endl;
	std::cout << "Executed program successfully." << std::endl;
	Cleanup(context, commandQueue, program, kernel, memObjects);

	delete [] arrDest;

	return 0;
}

long long Fact(long long n)
{
	long long fact = 1;
	for (long long i = 2; i <= n; ++i)
	{
		fact *= i;
	}

	return fact;
}

void check(char* arrDest, long long Max)
{
	std::cout << std::endl;
	std::cout << "Checking..." << std::endl;

	char check[NUMBER_OF_ELEMENTS];
	for(int i=0; i<NUMBER_OF_ELEMENTS ;++i)
	{
		check[i] = i;
	}

	if(OFFSET!=0)
	{
		for(int i=0; i<OFFSET; ++i)
		{
			std::next_permutation(check, check+NUMBER_OF_ELEMENTS);
		}
	}

	for(int i=0; i<Max ;++i)
	{
		for(int j=0;j<NUMBER_OF_ELEMENTS;++j)
		{
			if(arrDest[i*NUMBER_OF_ELEMENTS+j] != check[j])
			{
				printf("Diff check failed at %d!", i);
				return;
			}
		}

		std::next_permutation(check, check+NUMBER_OF_ELEMENTS);
	}

}

void display(char* arrDest, long long Max)
{
	for(int i=0; i<Max ;++i)
	{
		for(int j=0;j<NUMBER_OF_ELEMENTS;++j)
			std::cout << (int)(arrDest[i*NUMBER_OF_ELEMENTS+j]);

		std::cout << std::endl;
	}
}

bool InitMMTimer(UINT wTimerRes)
{
	TIMECAPS tc;

	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) 
	{
		return false;
	}

	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 

	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init)
{
	if(init)
		timeEndPeriod(wTimerRes);
}