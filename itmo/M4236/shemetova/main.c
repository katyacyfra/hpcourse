#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define MAX_SOURCE_SIZE (0x100000)
#define BLOCK (256)

int closest_size(int size) {
    return ceil(size/(float) BLOCK)*BLOCK;
}

void block_sum(float* in, float* B, const int N) {
    // Load the kernel source code into the array source_str


    //binding to BLOCK size
    float* A;
    int global = closest_size(N);

    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("prefix_summ.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1,
                          &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to get deviceID! %d\n", ret);
        exit(1);
    }

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      N * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      N * sizeof(float), NULL, &ret);

    // Copy the list in to respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                               N *  sizeof(float), in, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to copy buffers! %d\n", ret);
        exit(1);
    }


    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                         (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to build program! %d\n", ret);
        exit(1);
    }

    // load named kernel from opencl source
    cl_event event = NULL;

    cl_kernel kernel_hillis = clCreateKernel(program, "scan_hillis_steele", &ret);
    //https://stackoverflow.com/questions/43542618/how-to-pass-a-array-in-local-address-space-in-opencl-kernel/43581209
    ret = clSetKernelArg(kernel_hillis, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel_hillis, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel_hillis, 2, sizeof(float) * BLOCK, NULL),
    ret = clSetKernelArg(kernel_hillis, 3, sizeof(float) * BLOCK, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to set args! %d\n", ret);
        exit(1);
    }

    const size_t local = BLOCK;
    const size_t global_ = global;
    ret = clEnqueueNDRangeKernel(command_queue, kernel_hillis, 1, NULL,
                                 &global_, &local, 0, NULL, &event);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", ret);
        exit(1);
    }

    clWaitForEvents(1, &event);

    ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", ret);
        exit(1);
    }


    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel_hillis);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);


    if (N <= BLOCK) {
        return;

    }

    int group_count = 1 + ((N - 1) / BLOCK);


    float* g_sums = (float *)malloc(group_count * sizeof(float));
    for (int i = 0; i < group_count; ++i) {
        if ((i + 1) * BLOCK - 1 >= N) {
            g_sums[i] = B[N-1];
        }
        else {
            g_sums[i] = B[(i + 1) * BLOCK - 1];
        }
    }

    float* g_sums_result = (float *)malloc(group_count * sizeof(float));
    block_sum(g_sums,  g_sums_result, group_count);


    //////////////// AGGREGATION ////////////////////////////////
    // Create an OpenCL context
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1,
                          &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to build program! %d\n", ret);
        exit(1);
    }
    cl_context context_agg = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue_agg = clCreateCommandQueue(context_agg, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem in_mem_obj = clCreateBuffer(context_agg, CL_MEM_READ_ONLY,
                                       N * sizeof(float), NULL, &ret);
    cl_mem gr_mem_obj = clCreateBuffer(context_agg, CL_MEM_READ_ONLY,
                                       group_count * sizeof(float), NULL, &ret);
    cl_mem out_mem_obj = clCreateBuffer(context_agg, CL_MEM_WRITE_ONLY,
                                        N * sizeof(float), NULL, &ret);

    // Copy the list B and gr_sums to respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue_agg, in_mem_obj, CL_TRUE, 0,
                               N *  sizeof(float), B, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue_agg, gr_mem_obj, CL_TRUE, 0,
                               group_count *  sizeof(float), g_sums_result, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue_agg, out_mem_obj, CL_TRUE, 0,
                               group_count *  sizeof(float), B, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to copy buffers! %d\n", ret);
        exit(1);
    }

    // load named kernel from opencl source
    event = NULL;

    cl_kernel kernel_agg = clCreateKernel(program, "aggregate_sums", &ret);

    ret = clSetKernelArg(kernel_agg, 0, sizeof(cl_mem), (void *)&in_mem_obj);
    ret = clSetKernelArg(kernel_agg, 1, sizeof(cl_mem), (void *)&gr_mem_obj);
    ret = clSetKernelArg(kernel_agg, 2, sizeof(cl_mem), (void *)&out_mem_obj);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to set args! %d\n", ret);
        exit(1);
    }

    ret = clEnqueueNDRangeKernel(command_queue, kernel_agg, 1, NULL,
                                 &global_, &local, 0, NULL, &event);
    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to run! %d\n", ret);
        exit(1);
    }

    clWaitForEvents(1, &event);

    ret = clEnqueueReadBuffer(command_queue_agg, out_mem_obj, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);


    if (ret != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve result! %d\n", ret);
        exit(1);
    }

    ret = clFlush(command_queue_agg);
    ret = clFinish(command_queue_agg);
    ret = clReleaseKernel(kernel_agg);
    ret = clReleaseMemObject(in_mem_obj);
    ret = clReleaseMemObject(out_mem_obj);
    ret = clReleaseMemObject(gr_mem_obj);
    ret = clReleaseProgram(program);

    ret = clReleaseCommandQueue(command_queue_agg);
    ret = clReleaseContext(context_agg);
    free(A);
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Error: Please, specify input file!");
        exit(1);
    }

    // Initialize the input array
    FILE* file = fopen(argv[1], "r");

    if (file == NULL) {
        printf("Error Reading File\n");
        exit (0);
    }

    int N;
    int sc;

    sc = fscanf(file, "%d,", &N);


    float* A = (float *)malloc(N * sizeof(float));
    float* B = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        sc = fscanf(file, "%f", &A[i]);
    }


    fclose(file);

    //recursive count of sums
    block_sum(A, B, N);

    for(int j = 0; j < N; ++j) {
        printf("%.3f ", B[j]);
    }
    printf("\n");
    free(A);
    free(B);
    return 0;
}

