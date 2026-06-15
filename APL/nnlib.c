#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <float.h>  //max_pool
#define DEBUG_TIME
#define im2colxGEMM

struct tensor
{
    int n;
    int c;
    int h;
    int w;
    int dim;
    int size;
    float * data;
};

typedef struct tensor tensor;

tensor T[512];

// ------------------------------------------------------------
// Global OpenCL objects
// ------------------------------------------------------------
//int N,C,H,W;
//float y;
//float bias, epsilon;
//int padding, stride, groups, size;
static cl_platform_id gPlatform = NULL;
static cl_device_id gDevice = NULL;
static cl_context gContext = NULL;
static cl_command_queue gQueue = NULL;
static cl_program gProgram = NULL;
static int gOpenCLReady = 0;
cl_kernel k_bn_relu6_fused;  // 加這行

// ------------------------------------------------------------
// Initialize OpenCL runtime
// ------------------------------------------------------------
char* read_kernel_source(const char* filename, size_t* out_size)
{
    FILE* fp = fopen(filename, "rb");
    char* source = NULL;
    long file_size;

    if (!fp) {
        fprintf(stderr, "open kernel file failed: %s\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(fp);
        fprintf(stderr, "kernel file is empty: %s\n", filename);
        return NULL;
    }

    source = (char*)malloc((size_t)file_size + 1);
    if (!source) {
        fclose(fp);
        fprintf(stderr, "malloc failed for kernel source\n");
        return NULL;
    }

    if (fread(source, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fclose(fp);
        free(source);
        fprintf(stderr, "read kernel file failed: %s\n", filename);
        return NULL;
    }

    source[file_size] = '\0';
    fclose(fp);

    if (out_size) *out_size = (size_t)file_size;
    return source;
}

int init_opencl_common()
{
    if (gOpenCLReady) return 1;

    cl_int err;
    char* source = NULL;
    size_t source_size = 0;

    err = clGetPlatformIDs(1, &gPlatform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs failed: %d\n", err);
        return 0;
    }

    err = clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_GPU, 1, &gDevice, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs GPU failed: %d, fallback to CPU\n", err);
        err = clGetDeviceIDs(gPlatform, CL_DEVICE_TYPE_CPU, 1, &gDevice, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs CPU failed: %d\n", err);
            return 0;
        }
    }

    gContext = clCreateContext(NULL, 1, &gDevice, NULL, NULL, &err);
    if (err != CL_SUCCESS || gContext == NULL) {
        fprintf(stderr, "clCreateContext failed: %d\n", err);
        return 0;
    }

    gQueue = clCreateCommandQueue(gContext, gDevice, 0, &err);
    if (err != CL_SUCCESS || gQueue == NULL) {
        fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
        return 0;
    }

    source = read_kernel_source("../kernel.cl", &source_size);
    if (!source) {
        return 0;
    }

    gProgram = clCreateProgramWithSource(gContext, 1,
                                         (const char**)&source,
                                         &source_size, &err);
    free(source);

    if (err != CL_SUCCESS || gProgram == NULL) {
        fprintf(stderr, "clCreateProgramWithSource failed: %d\n", err);
        return 0;
    }

    err = clBuildProgram(gProgram, 1, &gDevice, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        char* log = NULL;

        clGetProgramBuildInfo(gProgram, gDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char*)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(gProgram, gDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
            free(log);
        } else {
            fprintf(stderr, "clBuildProgram failed and log malloc failed\n");
        }
        return 0;
    }

    printf("OpenCL init ... ok\n");
    gOpenCLReady = 1;
    return 1;
}

cl_kernel get_kernel(const char* name) {
    // Ensure OpenCL is initialized (context, program, etc.)
    if (!init_opencl_common()) return NULL;
    
    // Create kernel
    cl_int err;
    cl_kernel kernel = clCreateKernel(gProgram, name, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel(%s) failed: %d\n", name, err);
        return NULL;
    }
    return kernel;
}

float im2col_get_pixel(tensor * im, int height, int width, int channels,
        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
            row >= height || col >= width) return 0;

    return im->data[col + width * (row + height * channel)];
}

int where_pos2(tensor * out, int H, int W) 
{
    return H * out->w + W;
}

int where_pos4(tensor * out, int N, int C, int H, int W) 
{
    int hXw = out->h * out->w;
    return N * out->c * hXw + C * hXw + H * out->w + W;
}

tensor * make_tensor(tensor * out, int n, int c, int h, int w) 
{

    out->n = n;
    out->c = c;
    out->h = h;
    out->w = w;
    out->dim = 4;

    if (n == 0)
    {
        out->n = 1;
        out->dim = 3;
    }

    if (c == 0)
    {
        out->c = 1;
        out->dim = 2;
    }

    if (h == 0)
    {
        out->h = 1;
        out->dim = 1;
    }

    if (w == 0)
    {
        out->w = 1;
        out->dim = 0;
    }
    #define MALLOC_ALIGN    16
    out->size = out->n * out->c * out->h * out->w;
    if (!out->data) posix_memalign((void **)(&(out->data)), MALLOC_ALIGN, out->size  * sizeof(float));
    //if (!out->data) out->data = (float * ) calloc(out->size, sizeof(float));
    return out;
}

void variable(tensor * out, int n, int c, int h, int w, const char * label)
{
	FILE * pFile;
	out = make_tensor(out, n, c, h, w);

	char * file_name = (char *) calloc(strlen(label) + strlen(".dat") + 1, sizeof(char));
	sprintf(file_name, "%s%s", label, ".dat");
    pFile = fopen ( file_name , "rb" );

/*
    string file_name(label);
    file_name = file_name + ".dat";
	pFile = fopen ( file_name.c_str() , "rb" );
*/
	if (pFile==NULL) {fputs ("File error\n",stderr); assert (0);}

	fseek(pFile, 0, SEEK_END);
	int file_size = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);
	//printf("file_size %d\n", file_size);

	char metadata[4];
	fread((char *)metadata, sizeof(unsigned char), 4, pFile);
	unsigned char magic1 = metadata[0];
	unsigned char magic2 = metadata[1];
	unsigned char major  = metadata[2];
	unsigned char minor  = metadata[3];
	//printf("magic1 : %c\n", magic1);

	uint32_t data_length;
	fread((char *)&data_length, sizeof(uint32_t), 1, pFile);
	uint32_t rank;
	fread((char *)&rank, sizeof(uint32_t), 1, pFile);
	uint32_t ranks[4];
	uint32_t *shape = ranks;
	//printf("data_length : %d\n", data_length);
	//printf("rank : %d\n", rank);

	int checkVersion = 1;

	for (int i = 0; i < rank; ++i)
	{
		uint32_t *p = shape + i;
		fread((char *)p, sizeof(uint32_t), 1, pFile);
		checkVersion = checkVersion * (*p);
	}
	//printf("checkVersion : %d\n", checkVersion);

	int header_size = 128;
	int version = 0;
	if (file_size == header_size + data_length && (checkVersion * 4) == data_length)
	{
		assert((file_size - header_size) == (checkVersion * 4));
		//printf("Ver 2.0 - 60ba79d\n");
		version = 2;
	}
	else
	{
		//printf("Ver 1.0 - 02a3916\n");
		version = 1;
	}

	uint8_t code, bits;
	fread((char *)&code, sizeof(uint8_t), 1, pFile);
	fread((char *)&bits, sizeof(uint8_t), 1, pFile);
	uint16_t qlen;
	fread((char *)&qlen, sizeof(uint16_t), 1, pFile);

	uint64_t quantization;
	fread((char *)&quantization, sizeof(uint64_t), qlen, pFile);
	size_t count = rank ? 1 : 0;
	for (int i = 0; i < rank; ++i)
		count *= shape[i];

	//printf("count : %ld]\n", count);

	//NNEF store file format Version 2
	if (version == 2)
	{
		fseek(pFile, header_size, SEEK_SET);
		int result = fread (out->data, sizeof(float), out->size, pFile);
		if (result != out->size) {fputs ("Reading error\n",stderr); assert (0);}
	}
	else if (version == 1)
	{
		int result = fread(out->data, sizeof(float), count, pFile);
		if (result != count) {fputs ("Reading error\n",stderr); assert (0);}
	}
	/*
	   printf("file_name %s\n", file_name);
	   printf("out->n : %d\n", out->n);
	   printf("out->c : %d\n", out->c);
	   printf("out->h : %d\n", out->h);
	   printf("out->w : %d\n", out->w);
	   printf("out->dim : %d\n", out->dim);
	   printf("out->size : %d\n", out->size);
	   printf("out->data[0] : %g\n", out->data[0]);
	 */
	fclose(pFile);
}

void external(tensor * out, int n, int c, int h, int w)
{
    out = make_tensor(out, n, c, h, w);

    printf ("out->n : %d\n", out->n);
    printf ("out->c : %d\n", out->c);
    printf ("out->h : %d\n", out->h);
    printf ("out->w : %d\n", out->w);
    printf ("out->dim : %d\n", out->dim);
    printf ("out->size : %d\n", out->size);
}

void matmul(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    int m = in_x->h;
    int p = in_x->w;
    int n = in_y->w;

    out = make_tensor(out, 0, 0, m, n);

    if (!init_opencl_common()) {
        // fallback CPU
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < p; k++) {
                    sum += in_x->data[where_pos2(in_x, i, k)] *
                           in_y->data[where_pos2(in_y, k, j)];
                }
                out->data[where_pos2(out, i, j)] = sum;
            }
        }
#ifdef DEBUG_TIME
        double end = clock();
        printf("[matmul time = %1.3f seconds]\n", (end-start)/CLOCKS_PER_SEC);
#endif
        return;
    }

    cl_int err;
    cl_kernel kernel = NULL;
    cl_mem bufA = NULL, bufB = NULL, bufC = NULL;

    size_t bytesA = sizeof(float) * m * p;
    size_t bytesB = sizeof(float) * p * n;
    size_t bytesC = sizeof(float) * m * n;

    const size_t TS = 16;
    size_t local[2];
    size_t global[2];

    // kernel = get_kernel("matmul_tiled_kernel");
    // kernel = get_kernel("matmul_vec_kernel");
    kernel = get_kernel("matmul_tiled_vec_kernel");
    if (!kernel) goto cpu_fallback;

    bufA = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytesA, in_x->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    bufB = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytesB, in_y->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    bufC = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY,
                          bytesC, NULL, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &m);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &p);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &n);
    if (err != CL_SUCCESS) goto cpu_fallback;

    local[0] = TS;
    local[1] = TS;
    global[0] = ((size_t)n + TS - 1) / TS * TS;
    global[1] = ((size_t)m + TS - 1) / TS * TS;
    // global[0] = (size_t)n;
    // global[1] = (size_t)m;

    err = clEnqueueNDRangeKernel(gQueue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    // err = clEnqueueNDRangeKernel(gQueue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    err = clEnqueueReadBuffer(gQueue, bufC, CL_TRUE, 0, bytesC, out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    clReleaseMemObject(bufA); bufA = NULL;
    clReleaseMemObject(bufB); bufB = NULL;
    clReleaseMemObject(bufC); bufC = NULL;
    clReleaseKernel(kernel); kernel = NULL;

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[matmul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
    return;

cpu_fallback:
    if (bufA) clReleaseMemObject(bufA);
    if (bufB) clReleaseMemObject(bufB);
    if (bufC) clReleaseMemObject(bufC);
    if (kernel) clReleaseKernel(kernel);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < p; k++) {
                sum += in_x->data[where_pos2(in_x, i, k)] *
                       in_y->data[where_pos2(in_y, k, j)];
            }
            out->data[where_pos2(out, i, j)] = sum;
        }
    }

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[matmul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
}

void matmul_ft(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    int m = in_x->h;
    int p = in_x->w;
    int n = in_y->h;
    out = make_tensor(out, 0, 0, m, n);

    // [m*p][p*m] = [m*n]
    for (int i=0; i < m; i++)
    {
        for (int j=0; j < n; j++)
        {
            float sum = 0.0;
            for(int k = 0; k < p; k++)
            {
                sum += in_x->data[where_pos2(in_x, i, k)] * in_y->data[where_pos2(in_y, j, k)];
            }
            out->data[where_pos2(out, i, j)] = sum ;
        }
    }

#ifdef DEBUG_TIME
        double end = clock();
        printf("[matmul_ft time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void mul(tensor * out, tensor * in_x, float value)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    out->dim = in_x->dim;
    for (int i = 0; i < in_x->size; i++)
	out->data[i] = in_x->data[i] * value;

#ifdef DEBUG_TIME
        double end = clock();
        printf("[mul time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

// optional: try OpenCL, fallback to CPU
// but default keep CPU version
void add(tensor * out, tensor * in_x, tensor * in_y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    /* =========================
       填空 1：先嘗試初始化 OpenCL
       ========================= */
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = NULL;
    cl_mem buf_x = NULL;
    cl_mem buf_y = NULL;
    cl_mem buf_out = NULL;
    size_t global_size = 0;

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);
    global_size = (size_t)out->size;

    if (!init_opencl_common()) goto cpu_fallback;

    /* =========================
       填空 2：建立 input x buffer
       ========================= */
    buf_x = clCreateBuffer(gContext,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * in_x->size,
                           in_x->data,
                           &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    /* =========================
       填空 3：建立 output buffer
       ========================= */
    buf_out = clCreateBuffer(gContext,
                             CL_MEM_WRITE_ONLY,
                             sizeof(float) * out->size,
                             NULL,
                             &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    if (in_x->dim == in_y->dim)
    {
        assert(in_x->size == in_y->size);

        /* =========================
           填空 4：拿 same-shape kernel
           ========================= */
        kernel = get_kernel("add_same_shape");
        if (!kernel) goto cpu_fallback;

        buf_y = clCreateBuffer(gContext,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * in_y->size,
                               in_y->data,
                               &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        int total = out->size;

        /* =========================
           填空 5：設定 kernel args
           ========================= */
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_y);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &total);
        if (err != CL_SUCCESS) goto cpu_fallback;
    }
    else if ((in_x->dim == 4) && (in_y->dim == 2))
    {
        int N = in_x->n;
        int C = in_x->c;
        int H = in_x->h;
        int W = in_x->w;

        assert(in_y->size >= C);

        /* =========================
           填空 6：拿 bias-broadcast kernel
           ========================= */
        kernel = get_kernel("add_bias_4d");
        if (!kernel) goto cpu_fallback;

        buf_y = clCreateBuffer(gContext,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * in_y->size,
                               in_y->data,
                               &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        /* =========================
           填空 7：設定 kernel args
           ========================= */
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_y);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &C);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &H);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &W);
        if (err != CL_SUCCESS) goto cpu_fallback;
    }
    else
    {
        goto cpu_fallback;
    }

    /* =========================
       填空 8：launch kernel
       ========================= */
    err = clEnqueueNDRangeKernel(gQueue,
                                 kernel,
                                 1,
                                 NULL,
                                 &global_size,
                                 NULL,
                                 0,
                                 NULL,
                                 NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    /* =========================
       填空 9：read back
       ========================= */
    err = clEnqueueReadBuffer(gQueue,
                              buf_out,
                              CL_TRUE,
                              0,
                              sizeof(float) * out->size,
                              out->data,
                              0,
                              NULL,
                              NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    if (buf_x) clReleaseMemObject(buf_x);
    if (buf_y) clReleaseMemObject(buf_y);
    if (buf_out) clReleaseMemObject(buf_out);
    if (kernel) clReleaseKernel(kernel);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[add time = %1.3f seconds]\n", (end - start) / CLOCKS_PER_SEC);
    }
#endif
    return;

cpu_fallback:
    if (buf_x) clReleaseMemObject(buf_x);
    if (buf_y) clReleaseMemObject(buf_y);
    if (buf_out) clReleaseMemObject(buf_out);
    if (kernel) clReleaseKernel(kernel);

    if (in_x->dim == in_y->dim)
    {
        for (int i = 0; i < out->size; i++)
            out->data[i] = in_x->data[i] + in_y->data[i];
    }
    else if ((in_x->dim == 4) && (in_y->dim == 2))
    {
        for (int B = 0; B < in_x->n; B++)
            for (int C = 0; C < in_x->c; C++)
                for (int H = 0; H < in_x->h; H++)
                    for (int W = 0; W < in_x->w; W++)
                    {
                        out->data[where_pos4(out, B, C, H, W)] =
                            in_x->data[where_pos4(in_x, B, C, H, W)] + in_y->data[C];
                    }
    }
    else
    {
        printf("Unsupported tensor shape in add()\n");
        assert(0);
    }

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[add time = %1.3f seconds]\n", (end - start) / CLOCKS_PER_SEC);
    }
#endif
}

void softmax(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    cl_int err = CL_SUCCESS;
    cl_kernel k_exp = NULL;
    cl_kernel k_div = NULL;
    cl_mem bufX = NULL;
    cl_mem bufOut = NULL;

    int total = 0;
    size_t bytes = 0;
    size_t globalSize = 0;
    float sum = 0.0f;

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    total = out->size;
    bytes = (size_t)total * sizeof(float);
    globalSize = (size_t)total;

    if (!init_opencl_common()) goto fallback_cpu;

    k_exp = get_kernel("softmax_exp");
    if (!k_exp) goto fallback_cpu;

    bufX = clCreateBuffer(gContext,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytes,
                          in_x->data,
                          &err);
    if (err != CL_SUCCESS) goto fallback_cpu;

    bufOut = clCreateBuffer(gContext,
                            CL_MEM_READ_WRITE,
                            bytes,
                            NULL,
                            &err);
    if (err != CL_SUCCESS) goto fallback_cpu;

    err  = clSetKernelArg(k_exp, 0, sizeof(cl_mem), &bufX);
    err |= clSetKernelArg(k_exp, 1, sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(k_exp, 2, sizeof(int), &total);
    if (err != CL_SUCCESS) goto fallback_cpu;

    err = clEnqueueNDRangeKernel(gQueue, k_exp, 1, NULL,
                                 &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto fallback_cpu;

    err = clEnqueueReadBuffer(gQueue,
                              bufOut,
                              CL_TRUE,
                              0,
                              bytes,
                              out->data,
                              0,
                              NULL,
                              NULL);
    if (err != CL_SUCCESS) goto fallback_cpu;

    sum = 0.0f;
    for (int i = 0; i < total; i++)
        sum += out->data[i];

    assert(sum != 0.0f);

    k_div = get_kernel("softmax_div");
    if (!k_div) goto fallback_cpu;

    err  = clSetKernelArg(k_div, 0, sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(k_div, 1, sizeof(float), &sum);
    err |= clSetKernelArg(k_div, 2, sizeof(int), &total);
    if (err != CL_SUCCESS) goto fallback_cpu;

    err = clEnqueueNDRangeKernel(gQueue, k_div, 1, NULL,
                                 &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto fallback_cpu;

    err = clEnqueueReadBuffer(gQueue,
                              bufOut,
                              CL_TRUE,
                              0,
                              bytes,
                              out->data,
                              0,
                              NULL,
                              NULL);
    if (err != CL_SUCCESS) goto fallback_cpu;

    if (bufX) clReleaseMemObject(bufX);
    if (bufOut) clReleaseMemObject(bufOut);
    if (k_exp) clReleaseKernel(k_exp);
    if (k_div) clReleaseKernel(k_div);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[softmax time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
    return;

fallback_cpu:
    if (bufX) clReleaseMemObject(bufX);
    if (bufOut) clReleaseMemObject(bufOut);
    if (k_exp) clReleaseKernel(k_exp);
    if (k_div) clReleaseKernel(k_div);

    sum = 0.0f;
    for (int i = 0; i < total; i++) {
        out->data[i] = expf(in_x->data[i]);
        sum += out->data[i];
    }

    assert(sum != 0.0f);

    for (int i = 0; i < total; i++)
        out->data[i] /= sum;

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[softmax time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
}

//reshape = reshape(input_Placeholder, shape = [-1, 28, 28, 1]);
void reshape(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int shapeLength = 1;
    int tensorLength = 1;
    int index = -1;
    int shape[4];
    shape[0] = n;
    shape[1] = c;
    shape[2] = h;
    shape[3] = w;
    tensorLength = in_x->size;

    for (size_t i = 0; i < 4; ++i) 
    {
        if (shape[i] == -1) {
          // no -1 before
          assert(shapeLength > 0);
          index = i;
        }
        if (shape[i] == 0)
            shape[i] = 1;
        shapeLength *= shape[i];
    }

    if (shapeLength < 0) 
    {
        if (shapeLength != tensorLength) 
        {
            shapeLength = abs(shapeLength);
            shape[index] = tensorLength / shapeLength;
            shapeLength = tensorLength;
        }
    }

    assert(tensorLength == shapeLength);

    // Run
    out->n = shape[0];
    out->c = shape[1];
    out->h = shape[2];
    out->w = shape[3];

    if (n == 0)
    {
        out->n = 1;
        out->dim = 3;
    }

    if (c == 0)
    {
        out->c = 1;
        out->dim = 2;
    }

    if (h == 0)
    {
        out->h = 1;
        out->dim = 1;
    }

    if (w == 0)
    {
        out->w = 1;
        out->dim = 0;
    }

    for (int i = 0; i < out->size; i++)
        out->data[i] = in_x->data[i];

#ifdef DEBUG_TIME
    double end = clock();
    printf("[reshape time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}


void concat(tensor * out, tensor * in_x, tensor * in_y, int axis)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    //1-D
    //Chack
    assert (in_x->dim == in_y->dim);

    //Run
    out->n = in_x->n; out->c = in_x->c; out->h = in_x->h; out->w = in_x->w;

    if (axis == 0)
        out->n = out->n + in_y->n;
    else if (axis == 1)
        out->c = out->c + in_y->c;
    else if (axis == 2)
        out->n = out->n + in_y->h;
    else if (axis == 3)
        out->w = out->w + in_y->w;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // push size
    int push_size = 1;
    int run_size = 1;
    for (int i = in_x->dim - 1; i >=axis; i--)
    {
        int shap_num;
        if (i == 0)
            shap_num = in_x->n;
        else if (i == 1)
            shap_num = in_x->c;
        else if (i == 2)
            shap_num = in_x->h;
        else if (i == 3)
            shap_num = in_x->w;
        push_size = push_size * shap_num;
    }
    for (int i = axis - 1; i >=0; i--)
    {
        int shap_num;
        if (i == 0)
            shap_num = in_x->n;
        if (i == 1)
            shap_num = in_x->c;
        if (i == 2)
            shap_num = in_x->h;
        if (i == 3)
            shap_num = in_x->w;
        run_size = run_size * shap_num;
    }

    // total size check
    int cnt = 0;
    for (int i = 0; i < run_size; i++)
    {
        for (int j = 0; j < push_size; j++)
        {
            out->data[cnt] = in_x->data[i * push_size + j];
            cnt++;
        }

        for (int j = 0; j < push_size; j++)
        {
            out->data[cnt] = in_y->data[i * push_size + j];
            cnt++;
        }
    }
#ifdef DEBUG_TIME
    double end = clock();
    printf("[concat time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void squeeze(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int axes_size = 0;
    if (n != 0)
        axes_size++;
    if (c != 0)
        axes_size++;
    if (h != 0)
        axes_size++;
    if (w != 0)
        axes_size++;

    int count = 0;
    int cnt = 0;
    //run
    for (int i = 0; i < in_x->dim; i++)
    {
        if (count < axes_size)
        {
            int axes;
            if (count == 0)
                axes = n;
            if (count == 1)
                axes = c;
            if (count == 2)
                axes = h;
            if (count == 3)
                axes = w;
            if (i == axes)
            {
                count++;
                continue;
                //drop out
            }
        }

        int A_shape;
        if (i == 0)
            A_shape = in_x->n;
        else if (i == 1)
            A_shape = in_x->c;
        else if (i == 2)
            A_shape = in_x->h;
        else if (i == 3)
            A_shape = in_x->w;

        if (cnt == 0)
            out->n = A_shape;
        else if (cnt == 1)
            out->c = A_shape;
        else if (cnt == 2)
            out->h = A_shape;
        else if (cnt == 3)
            out->w = A_shape;
        cnt++;
        //out.shape.push_back(A.shape[i]);
    }

    for (int i = 0; i < in_x->size; i++)
        out->data[i] = in_x->data[i];

#ifdef DEBUG_TIME
    double end = clock();
    printf("[squeeze time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void transpose(tensor * out, tensor * in_x, int n, int c, int h, int w)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    int newA[4];
    int pos[4];
    int transPos[4];
    int shape[4];
    int in_x_shape[4];
    in_x_shape[0] = in_x->n;
    in_x_shape[1] = in_x->c;
    in_x_shape[2] = in_x->h;
    in_x_shape[3] = in_x->w;
    shape[0] = n;
    shape[1] = c;
    shape[2] = h;
    shape[3] = w;

    //newShape 
    out->n = in_x_shape[shape[0]]; 
    out->c = in_x_shape[shape[1]]; 
    out->h = in_x_shape[shape[2]]; 
    out->w = in_x_shape[shape[3]];     

    // init           
    for (int index = 0; index < 4; index++)
        pos[index] = 0;

    int S = 1;
    for (int index = 0; index < 4; index++)
        S = S * in_x_shape[index];

    for (int i = 0; i < S; i++)
    {
        int carryOut = 0;
        int index = i;
        // e.g 2 * 3 * 4, Loop of variable
        for (int j = 0; j < 4; j++)
        {   
            pos[j] = index % in_x_shape[j];
            carryOut = index / in_x_shape[j];
            index = carryOut;
        }

        for (int index = 0; index < 4; index++)
            transPos[index] = pos[shape[index]];

        out->data[where_pos4(out, transPos[0], transPos[1], transPos[2], transPos[3])]
        = in_x->data[where_pos4(in_x, pos[0], pos[1], pos[2], pos[3])];

    }

#ifdef DEBUG_TIME
    double end = clock();
    printf("[transpose time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void conv(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));
        
        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;
        
        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;
        
        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

#ifdef im2colxGEMM

    int out_w,out_h;
    int workspace_size;

    out_w = out->h;
    out_h = out->w;
    workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
    float * colD = 0;

    if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));    
    int c,h,w;

    int height_col = out_h;
    int width_col = out_w;
    int channels_col = in_x->c * filter->h * filter->h;

    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (c = 0; c < channels_col; ++c) 
        {
            for (h = 0; h < height_col; ++h) 
            {
                for (w = 0; w < width_col; ++w) 
                {
                    int w_offset = c % filter->h;
                    int h_offset = (c / filter->h) % filter->h;
                    int c_im = c / filter->h / filter->h;
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    //int col_index = (h * width_col + w) * channels_col + c;
                    colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
                }
            }
        }

        int m = filter->n; // input height N
        int n = out_w * out_h; // filter width = number of filter = 9
        int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

        for (int i=0; i < m; i++) //2
        {
            for (int j=0; j < n; j++) //9
            {
                float sum = 0.0;
                for(int k = 0; k < p; k++) //18
                {
                    sum += filter->data[i * p + k] * colD[k * n + j];
                }
                out->data[i*n+j] = sum + bias->data[i];
            }
        }

        free(colD);
    }
#else    
    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
        {
            for (int height = 0; height < out->h; height = height + 1)//28
            {
                for (int width = 0; width < out->w; width = width + 1)//28
                {
                    float featureValue = 0;
                    int offsetY = (height * stride);
                    int offsetX = (width  * stride);

                    for (int z = 0; z < filter->c; z++)
                    {
                        for (int y = 0; y < filter->h; y++)
                        {
                             for (int x = 0; x < filter->w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                    featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x];
                            }
                        }
                    }
                    out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias->data[filterKernel];
                }
            }
        }
    }
#endif    

#ifdef DEBUG_TIME
    double end = clock();
    printf("[conv time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}
//max_pool = max_pool(relu, border = 'constant', dilation = [], padding = [], size = [1, 1, 2, 2], stride = [1, 1, 2, 2]);
void max_pool(tensor * out, tensor * in_x, int size, int padding, int stride)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //Run
    //tensor<float> out;
    //out.shape.resize(4);

    //Chack
    assert(in_x->h >= size);
    assert(in_x->w >= size);

    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = (int)(ceil((float)(in_x->h)/(float)stride));
        out->w = (int)(ceil((float)(in_x->w)/(float)stride));

        int newY = size + (out->h - 1) * stride;
        int newX = size + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;
    }
    else
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = ceil(((float)(in_x->h - size + 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - size + 1))/((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out->n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out->c; C++)
            for (int H = 0; H < out->h; H++)
                for (int W = 0; W < out->w; W++)
                {
                    float MaxValue = -FLT_MAX;
                    int offsetY = (H  * stride);
                    int offsetX = (W  * stride);

                    //for (int x = 0; x < size[0]; x++)
                        //for (int y = 0; y < size[1]; y++)
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                            {
                                // logical_height, logical_weight
                                int l_height = z + offsetY;
                                int l_weight = t + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    float value = in_x->data[N * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)];
                                    if (MaxValue < value)
                                        MaxValue = value;
                                }
                            }
                    out->data[N * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W] = MaxValue;
                }

#ifdef DEBUG_TIME
    double end = clock();
    printf("[maxpool time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

// An OpenCL version of ReLU was implemented and tested, but not kept in the final submission
// because the host-device transfer and kernel launch overhead made it slower than the CPU baseline.
void relu(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    /* =========================
       先宣告（避免 goto 問題）
       ========================= */
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = NULL;
    cl_mem bufX = NULL;
    cl_mem bufOut = NULL;

    int total = out->size;
    size_t bytes = sizeof(float) * total;
    size_t globalSize = (size_t)total;

    /* =========================
       嘗試 OpenCL
       ========================= */
    if (!init_opencl_common()) goto cpu_fallback;

    kernel = get_kernel("relu");
    if (!kernel) goto cpu_fallback;

    bufX = clCreateBuffer(gContext,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytes,
                          in_x->data,
                          &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    bufOut = clCreateBuffer(gContext,
                            CL_MEM_WRITE_ONLY,
                            bytes,
                            NULL,
                            &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    /* =========================
       設定 kernel args
       ========================= */
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufX);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &total);
    if (err != CL_SUCCESS) goto cpu_fallback;

    /* =========================
       launch
       ========================= */
    err = clEnqueueNDRangeKernel(gQueue,
                                 kernel,
                                 1,
                                 NULL,
                                 &globalSize,
                                 NULL,
                                 0,
                                 NULL,
                                 NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    /* =========================
       read back
       ========================= */
    err = clEnqueueReadBuffer(gQueue,
                              bufOut,
                              CL_TRUE,
                              0,
                              bytes,
                              out->data,
                              0,
                              NULL,
                              NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    if (bufX) clReleaseMemObject(bufX);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[relu time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
    return;

cpu_fallback:
    if (bufX) clReleaseMemObject(bufX);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

    /* =========================
       CPU fallback（你原本那段）
       ========================= */
    for (int i = 0; i < out->size; i++)
        out->data[i] = fmax(in_x->data[i], 0.0f);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[relu time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
    }
#endif
}

void bn_relu6_fused(tensor* out, tensor* x,
                    tensor* mean, tensor* std,
                    tensor* gamma, tensor* beta)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, x->n, x->c, x->h, x->w);

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = NULL;
    cl_mem buf_x      = NULL;
    cl_mem buf_mean   = NULL;
    cl_mem buf_std = NULL;
    cl_mem buf_gamma  = NULL;
    cl_mem buf_beta   = NULL;
    cl_mem buf_out    = NULL;

    int HW    = x->h * x->w;
    int total = x->n * x->c * HW;
    size_t bytes_x = sizeof(float) * total;
    size_t bytes_c = sizeof(float) * x->c;
    size_t globalSize = (size_t)total;

    if (!init_opencl_common()) goto cpu_fallback;

    kernel = get_kernel("bn_relu6_fused");
    if (!kernel) goto cpu_fallback;

    buf_x = clCreateBuffer(gContext,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           bytes_x, x->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    buf_mean = clCreateBuffer(gContext,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              bytes_c, mean->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    buf_std = clCreateBuffer(gContext,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes_c, std->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    buf_gamma = clCreateBuffer(gContext,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytes_c, gamma->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    buf_beta = clCreateBuffer(gContext,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              bytes_c, beta->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    buf_out = clCreateBuffer(gContext,
                             CL_MEM_WRITE_ONLY,
                             bytes_x, NULL, &err);
    if (err != CL_SUCCESS) goto cpu_fallback;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_mean);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_std);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_gamma);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_beta);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_out);
    err |= clSetKernelArg(kernel, 6, sizeof(int),    &HW);
    if (err != CL_SUCCESS) goto cpu_fallback;

    err = clEnqueueNDRangeKernel(gQueue, kernel, 1,
                                 NULL, &globalSize, NULL,
                                 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    err = clEnqueueReadBuffer(gQueue, buf_out, CL_TRUE, 0,
                              bytes_x, out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback;

    if (buf_x)      clReleaseMemObject(buf_x);
    if (buf_mean)   clReleaseMemObject(buf_mean);
    if (buf_std)    clReleaseMemObject(buf_std);
    if (buf_gamma)  clReleaseMemObject(buf_gamma);
    if (buf_beta)   clReleaseMemObject(buf_beta);
    if (buf_out)    clReleaseMemObject(buf_out);
    if (kernel)     clReleaseKernel(kernel);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[bn_relu6_fused time = %1.3f seconds]\n", (end-start)/CLOCKS_PER_SEC);
    }
#endif
    return;

cpu_fallback:
    if (buf_x)      clReleaseMemObject(buf_x);
    if (buf_mean)   clReleaseMemObject(buf_mean);
    if (buf_std)    clReleaseMemObject(buf_std);
    if (buf_gamma)  clReleaseMemObject(buf_gamma);
    if (buf_beta)   clReleaseMemObject(buf_beta);
    if (buf_out)    clReleaseMemObject(buf_out);
    if (kernel)     clReleaseKernel(kernel);

    for (int b = 0; b < x->n; b++)
        for (int c = 0; c < x->c; c++)
            for (int hw = 0; hw < HW; hw++)
            {
                int idx = b * x->c * HW + c * HW + hw;
                float v = x->data[idx];
                v = (v - mean->data[c]) / std->data[c];
                v = v * gamma->data[c] + beta->data[c];
                v = fmaxf(0.0f, fminf(6.0f, v));
                out->data[idx] = v;
            }

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[bn_relu6_fused time = %1.3f seconds]\n", (end-start)/CLOCKS_PER_SEC);
    }
#endif
}

void convxbias_bn_relu6_fused(
    tensor* out,
    tensor* in_x,
    tensor* filter,
    float bias,
    tensor* mean,
    tensor* std,
    tensor* gamma,
    tensor* beta,
    int padding,
    int stride,
    int groups
)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));
        
        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;
        
        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;
        
        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

	if(groups == 1 && filter->h == 1 && filter->w == 1) //general convolution
	{
    cl_int err;
    cl_kernel kernel = NULL;
    cl_mem bufX      = NULL;
    cl_mem bufFilter = NULL;
    cl_mem bufMean   = NULL;  // ← 移到這裡
    cl_mem bufStd    = NULL;
    cl_mem bufGamma  = NULL;
    cl_mem bufBeta   = NULL;
    cl_mem bufOut    = NULL;
    int total = 0;
    size_t bytesX = 0;
    size_t bytesFilter = 0;
    size_t bytesOut = 0;
    size_t globalSize = 0;

    if (!init_opencl_common()) goto cpu_fallback_general;

    total = out->n * out->c * out->h * out->w;
    bytesX = sizeof(float) * in_x->n * in_x->c * in_x->h * in_x->w;
    bytesFilter = sizeof(float) * filter->n * filter->c * filter->h * filter->w;
    bytesOut = sizeof(float) * total;
    globalSize = (size_t)total;

    kernel = get_kernel("convxbias_bn_relu6_kernel");
    if (!kernel) goto cpu_fallback_general;

    bufX = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytesX, in_x->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;
    bufMean  = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * mean->size, mean->data, &err);
    bufStd   = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * std->size, std->data, &err);
    bufGamma = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * gamma->size, gamma->data, &err);
    bufBeta  = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * beta->size, beta->data, &err);

    bufFilter = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytesFilter, filter->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    bufOut = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY,
                            bytesOut, NULL, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err  = clSetKernelArg(kernel, 0,  sizeof(cl_mem), &bufX);
    err |= clSetKernelArg(kernel, 1,  sizeof(cl_mem), &bufFilter);
    err |= clSetKernelArg(kernel, 2,  sizeof(cl_mem), &bufMean);
    err |= clSetKernelArg(kernel, 3,  sizeof(cl_mem), &bufStd);
    err |= clSetKernelArg(kernel, 4,  sizeof(cl_mem), &bufGamma);
    err |= clSetKernelArg(kernel, 5,  sizeof(cl_mem), &bufBeta);
    err |= clSetKernelArg(kernel, 6,  sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(kernel, 7,  sizeof(float),  &bias);
    err |= clSetKernelArg(kernel, 8,  sizeof(int),    &in_x->n);
    err |= clSetKernelArg(kernel, 9,  sizeof(int),    &in_x->c);
    err |= clSetKernelArg(kernel, 10, sizeof(int),    &in_x->h);
    err |= clSetKernelArg(kernel, 11, sizeof(int),    &in_x->w);
    err |= clSetKernelArg(kernel, 12, sizeof(int),    &filter->n);
    err |= clSetKernelArg(kernel, 13, sizeof(int),    &filter->h);
    err |= clSetKernelArg(kernel, 14, sizeof(int),    &filter->w);
    err |= clSetKernelArg(kernel, 15, sizeof(int),    &out->c);
    err |= clSetKernelArg(kernel, 16, sizeof(int),    &out->h);
    err |= clSetKernelArg(kernel, 17, sizeof(int),    &out->w);
    err |= clSetKernelArg(kernel, 18, sizeof(int),    &stride);
    err |= clSetKernelArg(kernel, 19, sizeof(int),    &pad);
    err |= clSetKernelArg(kernel, 20, sizeof(int),    &total);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL,
                                 &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err = clEnqueueReadBuffer(gQueue, bufOut, CL_TRUE, 0,
                              bytesOut, out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    clReleaseMemObject(bufX);
    clReleaseMemObject(bufFilter);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kernel);

    goto done_general;

cpu_fallback_general:
    if (bufX) clReleaseMemObject(bufX);
    if (bufFilter) clReleaseMemObject(bufFilter);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)
        {
            for (int height = 0; height < out->h; height++)
            {
                for (int width = 0; width < out->w; width++)
                {
                    float featureValue = 0;
                    int offsetY = height * stride;
                    int offsetX = width * stride;

                    for (int z = 0; z < filter->c; z++)
                    {
                        for (int y = 0; y < filter->h; y++)
                        {
                            for (int x = 0; x < filter->w; x++)
                            {
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) &&
                                    (l_height < vb_height && l_weight < vb_width))
                                {
                                    featureValue +=
                                        in_x->data[Pic * in_x->c * in_x->h * in_x->w +
                                                   z * in_x->h * in_x->w +
                                                   (l_height - v_offset_Y) * in_x->w +
                                                   (l_weight - v_offset_X)]
                                        *
                                        filter->data[filterKernel * filter->c * filter->h * filter->w +
                                                     z * filter->h * filter->w +
                                                     y * filter->w + x];
                                }
                            }
                        }
                    }

                    out->data[Pic * out->c * out->h * out->w +
                              filterKernel * out->h * out->w +
                              height * out->w + width] = featureValue + bias;
                }
            }
        }
    }

    // ← 補這裡
    {
        int HW = out->h * out->w;
        for (int oc = 0; oc < out->c; oc++)
            for (int hw = 0; hw < HW; hw++)
            {
                int idx = oc * HW + hw;
                float v = out->data[idx];
                v = (v - mean->data[oc]) / std->data[oc];
                v = v * gamma->data[oc] + beta->data[oc];
                v = fmaxf(0.0f, fminf(6.0f, v));
                out->data[idx] = v;
            }
    }

done_general:
    ;
	}
    else if (groups == 1)
    {
#ifdef im2colxGEMM

		int out_w,out_h;
		int workspace_size;

		out_w = out->h;
		out_h = out->w;
		workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
		float * colD = 0;
		
		if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));    
		int c,h,w;

		int height_col = out_h;
		int width_col = out_w;
		int channels_col = in_x->c * filter->h * filter->h;
		
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (c = 0; c < channels_col; ++c) 
			{
				for (h = 0; h < height_col; ++h) 
				{
					for (w = 0; w < width_col; ++w) 
					{
						int w_offset = c % filter->h;
						int h_offset = (c / filter->h) % filter->h;
						int c_im = c / filter->h / filter->h;
						int im_row = h_offset + h * stride;
						int im_col = w_offset + w * stride;
						int col_index = (c * height_col + h) * width_col + w;
						//int col_index = (h * width_col + w) * channels_col + c;
						colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
					}
				}
			}

			int m = filter->n; // input height N
			int n = out_w * out_h; // filter width = number of filter = 9
			int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

			for (int i=0; i < m; i++) //2
			{
				for (int j=0; j < n; j++) //9
				{
					float sum = 0.0;
					for(int k = 0; k < p; k++) //18
					{
						// [ik][kj]
						sum += filter->data[i * p + k] * colD[k * n + j];
					}
					out->data[i*n+j] = sum + bias;
				}
			}

            // ← 補在這裡，GEMM 結束後套 BN+ReLU6
            int HW = out->h * out->w;
            for (int oc = 0; oc < out->c; oc++)
            {
                for (int hw = 0; hw < HW; hw++)
                {
                    int idx = oc * HW + hw;
                    float v = out->data[idx];
                    v = (v - mean->data[oc]) / std->data[oc];
                    v = v * gamma->data[oc] + beta->data[oc];
                    v = fmaxf(0.0f, fminf(6.0f, v));
                    out->data[idx] = v;
                }
            }

			free(colD);
		}
#else    
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

						for (int z = 0; z < filter->c; z++)
						{
							for (int y = 0; y < filter->h; y++)
							{
								 for (int x = 0; x < filter->w; x++)
								 {
									// logical_height, logical_weight
									int l_height = y + offsetY;
									int l_weight = x + offsetX;

									if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
										featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x];
								}
							}
						}
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}

        // ← 補這裡
        {
            int HW = out->h * out->w;
            for (int oc = 0; oc < out->c; oc++)
                for (int hw = 0; hw < HW; hw++)
                {
                    int idx = oc * HW + hw;
                    float v = out->data[idx];
                    v = (v - mean->data[oc]) / std->data[oc];
                    v = v * gamma->data[oc] + beta->data[oc];
                    v = fmaxf(0.0f, fminf(6.0f, v));
                    out->data[idx] = v;
                }
        }
#endif
    }
	else
	{
        int count = 0;
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

                        for (int y = 0; y < filter->h; y++)
                        {
                             for (int x = 0; x < filter->w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //colD[count] = in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)]; 
                                    //colD[count] = colD[count] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //featureValue = featureValue + colD[count];                                      
                                }
                            }
                        }
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}

        // ← 補在這裡，GEMM 結束後套 BN+ReLU6
        int HW = out->h * out->w;
        for (int oc = 0; oc < out->c; oc++)
        {
            for (int hw = 0; hw < HW; hw++)
            {
                int idx = oc * HW + hw;
                float v = out->data[idx];
                v = (v - mean->data[oc]) / std->data[oc];
                v = v * gamma->data[oc] + beta->data[oc];
                v = fmaxf(0.0f, fminf(6.0f, v));
                out->data[idx] = v;
            }
        }
	}
#ifdef DEBUG_TIME
    double end = clock();
    printf("[convxbias_bn_relu6_fused time = %1.3f seconds, groups = %d]\n",(end-start)/CLOCKS_PER_SEC, groups);
#endif
}

void sigmoid(tensor * out, tensor * in_x)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    for (int i = 0; i < out->size; i++)
        out->data[i] = 1.0 / (1.0 + expf(-in_x->data[i]));

#ifdef DEBUG_TIME
    double end = clock();
    printf("[sigmoid time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void convxbias(tensor * out, tensor * in_x, tensor * filter, float bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));
        
        //padding
        int newY = filter->h + (out->h - 1) * stride;
        int newX = filter->w + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width  = in_x->w + v_offset_X;
        
        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h+ 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w+ 1))/((float)stride));

        vb_height = in_x->h;
        vb_width  = in_x->w;
        
        pad = 0;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

	if(groups == 1 && filter->h == 1 && filter->w == 1) //general convolution
	{
    cl_int err;
    cl_kernel kernel = NULL;
    cl_mem bufX = NULL;
    cl_mem bufFilter = NULL;
    cl_mem bufOut = NULL;
    int total = 0;
    size_t bytesX = 0;
    size_t bytesFilter = 0;
    size_t bytesOut = 0;
    size_t globalSize = 0;

    if (!init_opencl_common()) goto cpu_fallback_general;

    total = out->n * out->c * out->h * out->w;
    bytesX = sizeof(float) * in_x->n * in_x->c * in_x->h * in_x->w;
    bytesFilter = sizeof(float) * filter->n * filter->c * filter->h * filter->w;
    bytesOut = sizeof(float) * total;
    globalSize = (size_t)total;

    kernel = get_kernel("convxbias_kernel");
    if (!kernel) goto cpu_fallback_general;

    bufX = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          bytesX, in_x->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    bufFilter = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytesFilter, filter->data, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    bufOut = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY,
                            bytesOut, NULL, &err);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufX);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufFilter);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &bias);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &in_x->n);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &in_x->c);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &in_x->h);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &in_x->w);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &filter->n);
    err |= clSetKernelArg(kernel, 9, sizeof(int), &filter->c);
    err |= clSetKernelArg(kernel, 10, sizeof(int), &filter->h);
    err |= clSetKernelArg(kernel, 11, sizeof(int), &filter->w);
    err |= clSetKernelArg(kernel, 12, sizeof(int), &out->c);
    err |= clSetKernelArg(kernel, 13, sizeof(int), &out->h);
    err |= clSetKernelArg(kernel, 14, sizeof(int), &out->w);
    err |= clSetKernelArg(kernel, 15, sizeof(int), &stride);
    err |= clSetKernelArg(kernel, 16, sizeof(int), &pad);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL,
                                 &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    err = clEnqueueReadBuffer(gQueue, bufOut, CL_TRUE, 0,
                              bytesOut, out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) goto cpu_fallback_general;

    clReleaseMemObject(bufX);
    clReleaseMemObject(bufFilter);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kernel);

    goto done_general;

cpu_fallback_general:
    if (bufX) clReleaseMemObject(bufX);
    if (bufFilter) clReleaseMemObject(bufFilter);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

    for (int Pic = 0; Pic < inPic; Pic++)
    {
        for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)
        {
            for (int height = 0; height < out->h; height++)
            {
                for (int width = 0; width < out->w; width++)
                {
                    float featureValue = 0;
                    int offsetY = height * stride;
                    int offsetX = width * stride;

                    for (int z = 0; z < filter->c; z++)
                    {
                        for (int y = 0; y < filter->h; y++)
                        {
                            for (int x = 0; x < filter->w; x++)
                            {
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) &&
                                    (l_height < vb_height && l_weight < vb_width))
                                {
                                    featureValue +=
                                        in_x->data[Pic * in_x->c * in_x->h * in_x->w +
                                                   z * in_x->h * in_x->w +
                                                   (l_height - v_offset_Y) * in_x->w +
                                                   (l_weight - v_offset_X)]
                                        *
                                        filter->data[filterKernel * filter->c * filter->h * filter->w +
                                                     z * filter->h * filter->w +
                                                     y * filter->w + x];
                                }
                            }
                        }
                    }

                    out->data[Pic * out->c * out->h * out->w +
                              filterKernel * out->h * out->w +
                              height * out->w + width] = featureValue + bias;
                }
            }
        }
    }

done_general:
    ;
	}
    else if (groups == 1)
    {
#ifdef im2colxGEMM

		int out_w,out_h;
		int workspace_size;

		out_w = out->h;
		out_h = out->w;
		workspace_size = out_h * out_w * filter->h * filter->h * in_x->c;
		float * colD = 0;
		
		if (!colD) colD = (float *) calloc(workspace_size, sizeof(float));    
		int c,h,w;

		int height_col = out_h;
		int width_col = out_w;
		int channels_col = in_x->c * filter->h * filter->h;
		
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (c = 0; c < channels_col; ++c) 
			{
				for (h = 0; h < height_col; ++h) 
				{
					for (w = 0; w < width_col; ++w) 
					{
						int w_offset = c % filter->h;
						int h_offset = (c / filter->h) % filter->h;
						int c_im = c / filter->h / filter->h;
						int im_row = h_offset + h * stride;
						int im_col = w_offset + w * stride;
						int col_index = (c * height_col + h) * width_col + w;
						//int col_index = (h * width_col + w) * channels_col + c;
						colD[col_index] = im2col_get_pixel(in_x , in_x->h, in_x->w, in_x->c, im_row, im_col, c_im, pad);
					}
				}
			}

			int m = filter->n; // input height N
			int n = out_w * out_h; // filter width = number of filter = 9
			int p = filter->c * filter->h * filter->w; // CHW = input width = filter height = channel*ksize*ksize

			for (int i=0; i < m; i++) //2
			{
				for (int j=0; j < n; j++) //9
				{
					float sum = 0.0;
					for(int k = 0; k < p; k++) //18
					{
						// [ik][kj]
						sum += filter->data[i * p + k] * colD[k * n + j];
					}
					out->data[i*n+j] = sum + bias;
				}
			}

			free(colD);
		}
#else    
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

						for (int z = 0; z < filter->c; z++)
						{
							for (int y = 0; y < filter->h; y++)
							{
								 for (int x = 0; x < filter->w; x++)
								 {
									// logical_height, logical_weight
									int l_height = y + offsetY;
									int l_weight = x + offsetX;

									if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
										featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + z * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + z * filter->h * filter->w + y * filter->w + x];
								}
							}
						}
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}
#endif
    }
	else
	{
        int count = 0;
		for (int Pic = 0; Pic < inPic; Pic++)
		{
			for (int filterKernel = 0; filterKernel < filterKernelNum; filterKernel++)// 32
			{
				for (int height = 0; height < out->h; height = height + 1)//28
				{
					for (int width = 0; width < out->w; width = width + 1)//28
					{
						float featureValue = 0;
						int offsetY = (height * stride);
						int offsetX = (width  * stride);

                        for (int y = 0; y < filter->h; y++)
                        {
                             for (int x = 0; x < filter->w; x++)
                             {
                                // logical_height, logical_weight
                                int l_height = y + offsetY;
                                int l_weight = x + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    featureValue = featureValue + in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //colD[count] = in_x->data[Pic * in_x->c * in_x->h * in_x->w + filterKernel * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)]; 
                                    //colD[count] = colD[count] * filter->data[filterKernel * filter->c * filter->h * filter->w + 0 * filter->h * filter->w + y * filter->w + x];
                                    //featureValue = featureValue + colD[count];                                      
                                }
                            }
                        }
						out->data[Pic * out->c * out->h * out->w + filterKernel * out->h * out->w + height * out->w + width] = featureValue + bias;
					}
				}
			}
		}
	}
#ifdef DEBUG_TIME
    double end = clock();
    printf("[convxbias time = %1.3f seconds, groups = %d]\n",(end-start)/CLOCKS_PER_SEC, groups);
#endif
}

void convxt_bias(tensor * out, tensor * in_x, tensor * filter, tensor * bias, int padding, int stride, int groups)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //shape
    int inPic = in_x->n;
    int filterKernelNum = filter->n;

    assert(in_x->h >= filter->h);
    assert(in_x->w >= filter->w);

    int pad = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)in_x->h)/((float)stride));
        out->w = ceil(((float)in_x->w)/((float)stride));

        pad = ((out->h - 1) * stride + filter->h - in_x->h) / 2;
    }
    else
    {
        out->n = in_x->n;
        out->c = filter->n;
        out->h = ceil(((float)(in_x->h - filter->h + 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - filter->w + 1))/((float)stride));

        pad = 0;
    }

    make_tensor(out, out->n, out->c, out->h, out->w);

    if (groups == 1)
    {
        // ----------------------------------------------------------------
        // im2col dimensions
        // ----------------------------------------------------------------
        int out_h        = out->h;
        int out_w        = out->w;
        int height_col   = out_h;
        int width_col    = out_w;
        int channels_col = in_x->c * filter->h * filter->w;
        int workspace_size = channels_col * height_col * width_col;

        int m = filter->n;                          // GEMM rows
        int n = out_h * out_w;                      // GEMM cols
        int p = filter->c * filter->h * filter->w;  // GEMM inner dim

        // ----------------------------------------------------------------
        // OpenCL buffers / kernels
        // ----------------------------------------------------------------
        cl_int   err       = CL_SUCCESS;
        cl_kernel k_im2col = NULL;
        cl_kernel k_gemm   = NULL;
        cl_mem   buf_in    = NULL;
        cl_mem   buf_filter= NULL;
        cl_mem   buf_bias  = NULL;
        cl_mem   buf_colD  = NULL;
        cl_mem   buf_out   = NULL;

        size_t bytes_in     = (size_t)in_x->n * in_x->c * in_x->h * in_x->w * sizeof(float);
        size_t bytes_filter = (size_t)filter->n * filter->c * filter->h * filter->w * sizeof(float);
        size_t bytes_bias   = (size_t)filter->n * sizeof(float);
        size_t bytes_colD   = (size_t)workspace_size * sizeof(float);
        size_t bytes_out    = (size_t)out->n * out->c * out->h * out->w * sizeof(float);

        int in_plane  = in_x->c * in_x->h * in_x->w;
        int out_plane = out->c  * out->h  * out->w;

        const size_t TS = 16;

        if (!init_opencl_common()) goto cpu_fallback;

        k_im2col = get_kernel("im2col_kernel");
        if (!k_im2col) goto cpu_fallback;

        k_gemm = get_kernel("gemm_bias_kernel");
        if (!k_gemm) goto cpu_fallback;

        buf_in = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes_in, in_x->data, &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        buf_filter = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    bytes_filter, filter->data, &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        buf_bias = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes_bias, bias->data, &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        buf_colD = clCreateBuffer(gContext, CL_MEM_READ_WRITE,
                                  bytes_colD, NULL, &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        buf_out = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY,
                                 bytes_out, NULL, &err);
        if (err != CL_SUCCESS) goto cpu_fallback;

        for (int Pic = 0; Pic < inPic; Pic++)
        {
            int in_offset  = Pic * in_plane;
            int out_offset = Pic * out_plane;

            // ---- im2col ----
            err  = clSetKernelArg(k_im2col,  0, sizeof(cl_mem), &buf_in);
            err |= clSetKernelArg(k_im2col,  1, sizeof(cl_mem), &buf_colD);
            err |= clSetKernelArg(k_im2col,  2, sizeof(int),    &in_x->h);
            err |= clSetKernelArg(k_im2col,  3, sizeof(int),    &in_x->w);
            err |= clSetKernelArg(k_im2col,  4, sizeof(int),    &in_x->c);
            err |= clSetKernelArg(k_im2col,  5, sizeof(int),    &filter->h);
            err |= clSetKernelArg(k_im2col,  6, sizeof(int),    &filter->w);
            err |= clSetKernelArg(k_im2col,  7, sizeof(int),    &out_h);
            err |= clSetKernelArg(k_im2col,  8, sizeof(int),    &out_w);
            err |= clSetKernelArg(k_im2col,  9, sizeof(int),    &stride);
            err |= clSetKernelArg(k_im2col, 10, sizeof(int),    &pad);
            err |= clSetKernelArg(k_im2col, 11, sizeof(int),    &in_offset);
            if (err != CL_SUCCESS) goto cpu_fallback;

            size_t im2col_global[3] = {
                (size_t)channels_col,
                (size_t)height_col,
                (size_t)width_col
            };
            err = clEnqueueNDRangeKernel(gQueue, k_im2col, 3, NULL,
                                         im2col_global, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) goto cpu_fallback;

            // ---- GEMM + bias ----
            err  = clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &buf_filter);
            err |= clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &buf_colD);
            err |= clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &buf_bias);
            err |= clSetKernelArg(k_gemm, 3, sizeof(cl_mem), &buf_out);
            err |= clSetKernelArg(k_gemm, 4, sizeof(int),    &m);
            err |= clSetKernelArg(k_gemm, 5, sizeof(int),    &n);
            err |= clSetKernelArg(k_gemm, 6, sizeof(int),    &p);
            err |= clSetKernelArg(k_gemm, 7, sizeof(int),    &out_offset);
            if (err != CL_SUCCESS) goto cpu_fallback;

            size_t local[2]  = { TS, TS };
            size_t global[2] = {
                ((size_t)m + TS - 1) / TS * TS,
                ((size_t)n + TS - 1) / TS * TS
            };
            err = clEnqueueNDRangeKernel(gQueue, k_gemm, 2, NULL,
                                         global, local, 0, NULL, NULL);
            if (err != CL_SUCCESS) goto cpu_fallback;
        }

        err = clEnqueueReadBuffer(gQueue, buf_out, CL_TRUE, 0,
                                  bytes_out, out->data, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto cpu_fallback;

        if (buf_in)     clReleaseMemObject(buf_in);
        if (buf_filter) clReleaseMemObject(buf_filter);
        if (buf_bias)   clReleaseMemObject(buf_bias);
        if (buf_colD)   clReleaseMemObject(buf_colD);
        if (buf_out)    clReleaseMemObject(buf_out);
        if (k_im2col)   clReleaseKernel(k_im2col);
        if (k_gemm)     clReleaseKernel(k_gemm);

        goto done;

cpu_fallback:
        if (buf_in)     clReleaseMemObject(buf_in);
        if (buf_filter) clReleaseMemObject(buf_filter);
        if (buf_bias)   clReleaseMemObject(buf_bias);
        if (buf_colD)   clReleaseMemObject(buf_colD);
        if (buf_out)    clReleaseMemObject(buf_out);
        if (k_im2col)   clReleaseKernel(k_im2col);
        if (k_gemm)     clReleaseKernel(k_gemm);

        // ---- CPU fallback: im2col + GEMM ----
        {
            float *colD = (float *)calloc(workspace_size, sizeof(float));

            for (int Pic = 0; Pic < inPic; Pic++)
            {
                // im2col
                for (int c = 0; c < channels_col; ++c)
                    for (int h = 0; h < height_col; ++h)
                        for (int w = 0; w < width_col; ++w)
                        {
                            int w_offset  = c % filter->w;
                            int h_offset  = (c / filter->w) % filter->h;
                            int c_im      = c / filter->w / filter->h;
                            int im_row    = h_offset + h * stride;
                            int im_col    = w_offset + w * stride;
                            int col_index = (c * height_col + h) * width_col + w;
                            colD[col_index] = im2col_get_pixel(in_x, in_x->h, in_x->w,
                                                               in_x->c, im_row, im_col,
                                                               c_im, pad);
                        }

                // GEMM + bias
                int out_offset = Pic * out_plane;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < p; k++)
                            sum += filter->data[i * p + k] * colD[k * n + j];
                        out->data[out_offset + i * n + j] = sum + bias->data[i];
                    }
            }
            free(colD);
        }

done:
        ;
    }
    else
    {
        assert(0);
    }

#ifdef DEBUG_TIME
    double end = clock();
    printf("[convxbias time = %1.3f seconds, groups = %d]\n",(end-start)/CLOCKS_PER_SEC, groups);
#endif
}
void batch_normalization(tensor * out, tensor * in_x, tensor * mean, tensor * variance, tensor * offset, tensor * scale, float epsilon)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    cl_int err;
    cl_mem bufX = NULL;
    cl_mem bufMean = NULL;
    cl_mem bufVar = NULL;
    cl_mem bufOffset = NULL;
    cl_mem bufScale = NULL;
    cl_mem bufOut = NULL;
    cl_kernel kernel = NULL;

    int total;
    int n, c, h, w;
    size_t bytesX, bytesC, globalSize;

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    if (!init_opencl_common()) {
        fprintf(stderr, "OpenCL init failed, fallback to CPU batch_normalization\n");
        goto fallback_cpu;
    }

    total = out->size;
    n = in_x->n;
    c = in_x->c;
    h = in_x->h;
    w = in_x->w;

    bytesX = (size_t)total * sizeof(float);
    bytesC = (size_t)c * sizeof(float);
    globalSize = (size_t)total;

    kernel = get_kernel("batch_normalization");
    if (!kernel) goto fallback_cpu;

    bufX = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesX, in_x->data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufX failed: %d\n", err);
        goto fallback_cpu;
    }

    bufMean = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesC, mean->data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufMean failed: %d\n", err);
        goto fallback_cpu;
    }

    bufVar = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesC, variance->data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufVar failed: %d\n", err);
        goto fallback_cpu;
    }

    bufOffset = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesC, offset->data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufOffset failed: %d\n", err);
        goto fallback_cpu;
    }

    bufScale = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesC, scale->data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufScale failed: %d\n", err);
        goto fallback_cpu;
    }

    bufOut = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY, bytesX, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer bufOut failed: %d\n", err);
        goto fallback_cpu;
    }

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufX);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufMean);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufVar);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufOffset);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufScale);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufOut);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &n);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &c);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &h);
    err |= clSetKernelArg(kernel, 9, sizeof(int), &w);
    err |= clSetKernelArg(kernel, 10, sizeof(float), &epsilon);
    err |= clSetKernelArg(kernel, 11, sizeof(int), &total);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg batch_normalization failed: %d\n", err);
        goto fallback_cpu;
    }

    err = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel batch_normalization failed: %d\n", err);
        goto fallback_cpu;
    }

    err = clEnqueueReadBuffer(gQueue, bufOut, CL_TRUE, 0, bytesX, out->data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer batch_normalization failed: %d\n", err);
        goto fallback_cpu;
    }

    if (bufX) clReleaseMemObject(bufX);
    if (bufMean) clReleaseMemObject(bufMean);
    if (bufVar) clReleaseMemObject(bufVar);
    if (bufOffset) clReleaseMemObject(bufOffset);
    if (bufScale) clReleaseMemObject(bufScale);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[batch_normalization time = %1.3f seconds]\n", (end-start)/CLOCKS_PER_SEC);
    }
#endif
    return;

fallback_cpu:
    if (bufX) clReleaseMemObject(bufX);
    if (bufMean) clReleaseMemObject(bufMean);
    if (bufVar) clReleaseMemObject(bufVar);
    if (bufOffset) clReleaseMemObject(bufOffset);
    if (bufScale) clReleaseMemObject(bufScale);
    if (bufOut) clReleaseMemObject(bufOut);
    if (kernel) clReleaseKernel(kernel);

    fprintf(stderr, "batch_normalization OpenCL failed, fallback to CPU\n");

    for (int B = 0; B < in_x->n; B++)
        for (int C = 0; C < in_x->c; C++)
            for (int H = 0; H < in_x->h; H++)
                for (int W = 0; W < in_x->w; W++)
                {
                    int out_index = B * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W;
                    int in_index  = B * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + H * in_x->w + W;
                    out->data[out_index] = (in_x->data[in_index] - mean->data[C]) / sqrt(variance->data[C] + epsilon);
                    out->data[out_index] = scale->data[C] * out->data[out_index] + offset->data[C];
                }

#ifdef DEBUG_TIME
    {
        double end = clock();
        printf("[batch_normalization time = %1.3f seconds]\n", (end-start)/CLOCKS_PER_SEC);
    }
#endif
}

void avg_pool(tensor * out, tensor * in_x, int size, int padding, int stride)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif
    //Run
    //tensor<float> out;
    //out.shape.resize(4);

    //Check
    assert(in_x->h >= size);
    assert(in_x->w >= size);

    int v_offset_T = 0;
    int v_offset_Z = 0;
    int v_offset_Y = 0;
    int v_offset_X = 0;

    //virtual_height, virtual_weight
    int v_height = 0;
    int v_width = 0;

    //virtual_bound_height , virtual_bound_weight
    int vb_height = 0;
    int vb_width = 0;

    if (padding)
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = (int)(ceil((float)(in_x->h)/(float)stride));
        out->w = (int)(ceil((float)(in_x->w)/(float)stride));

        int newY = size + (out->h - 1) * stride;
        int newX = size + (out->w - 1) * stride;

        v_offset_Y = (newY - in_x->h) / 2;
        v_offset_X = (newX - in_x->w) / 2;

        vb_height = in_x->h + v_offset_Y;
        vb_width = in_x->w + v_offset_X;
    }
    else
    {
        out->n = in_x->n;
        out->c = in_x->c;
        out->h = ceil(((float)(in_x->h - size + 1))/((float)stride));
        out->w = ceil(((float)(in_x->w - size + 1))/((float)stride));

        vb_height = in_x->h;
        vb_width = in_x->w;
    }

    //virtual_height, virtual_weight
    v_height = v_offset_Y;
    v_width = v_offset_X;

    make_tensor(out, out->n, out->c, out->h, out->w);

    // Tensor is [batch, height, width, channels], NNEF not
    // NNEF is [batch, channels, height, width]
    for (int N = 0; N < out->n; N++)
        //#pragma omp parallel for
        for (int C = 0; C < out->c; C++)
            for (int H = 0; H < out->h; H++)
                for (int W = 0; W < out->w; W++)
                {
                    float AvgValue = 0.0;
                    int Div = 0;
                    int offsetY = (H  * stride);
                    int offsetX = (W  * stride);
					int out_index = N * out->c * out->h * out->w + C * out->h * out->w + H * out->w + W;
					out->data[out_index] = 0;
                    //for (int x = 0; x < size[0]; x++)
                        //for (int y = 0; y < size[1]; y++)
                    for (int z = 0; z < size; z++)
                        for (int t = 0; t < size; t++)
                            {
                                // logical_height, logical_weight
                                int l_height = z + offsetY;
                                int l_weight = t + offsetX;

                                if ((l_height >= v_height && l_weight >= v_width) && (l_height < vb_height && l_weight < vb_width))
                                {
                                    float value = in_x->data[N * in_x->c * in_x->h * in_x->w + C * in_x->h * in_x->w + (l_height - v_offset_Y) * in_x->w + (l_weight - v_offset_X)];
                                    AvgValue = AvgValue + value;
                                }
                            }
                            out->data[out_index] = AvgValue / (size * size);

                }


#ifdef DEBUG_TIME
    double end = clock();
    printf("[avgpool time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void min(tensor * out, tensor * in_x, float y)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

	float tmp;
    for (int i = 0; i < out->size; i++)
	{
        tmp = in_x->data[i];
		out->data[i] = (tmp > y) ? y : tmp;
	}

#ifdef DEBUG_TIME
        double end = clock();
        printf("[min time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}

void bn_sqrt(tensor * out, tensor * in_x, float epsilon)
{
#ifdef DEBUG_TIME
    double start = clock();
#endif

    out = make_tensor(out, in_x->n, in_x->c, in_x->h, in_x->w);

    for (int i = 0; i < in_x->size; i++)
    {
        out->data[i] = sqrt(in_x->data[i] + epsilon);
    }

#ifdef DEBUG_TIME
        double end = clock();
        printf("[bn_sqrt time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
#endif
}
