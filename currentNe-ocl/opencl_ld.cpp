#include "opencl_ld.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace
{
void check(cl_int err, const char* where)
{
    if (err == CL_SUCCESS) return;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "OpenCL error %d at %s", err, where);
    throw std::runtime_error(buf);
}

std::string readTextFile(const std::string& filePath)
{
    std::ifstream input(filePath, std::ios::binary);
    if (!input.good()) return "";
    input.seekg(0, std::ios::end);
    std::streamoff length = input.tellg();
    if (length <= 0) return "";
    input.seekg(0, std::ios::beg);
    std::string data(static_cast<size_t>(length), '\0');
    input.read(&data[0], length);
    return data;
}

cl_mem makeZeroedBuffer(cl_context ctx, cl_command_queue queue, size_t nbytes)
{
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nbytes, nullptr, &err);
    check(err, "clCreateBuffer(out)");
    const cl_ulong zero = 0;
    err = clEnqueueFillBuffer(queue, buffer, &zero, sizeof(zero), 0, nbytes, 0, nullptr, nullptr);
    check(err, "clEnqueueFillBuffer(out)");
    return buffer;
}
} // namespace

void ComputeLD_OpenCL(
    const char* genoT, int N, int L,
    const char* cromo, const double* posiCM,
    bool flag_chr, bool flag_cM, double z_cm,
    long int* x_contapares, long int* x_containdX, double* xD, double* xW,
    long int* x_contapares05, long int* x_containdX05, double* xD05, double* xW05,
    long int* x_contapareslink, long int* x_containdXlink, double* xDlink, double* xWlink)
{
    if (genoT == nullptr || N <= 0 || L <= 1) return;

    cl_int err = CL_SUCCESS;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::vector<cl_mem> toRelease;

    auto cleanup = [&]() {
        for (cl_mem mem : toRelease)
        {
            if (mem != nullptr) clReleaseMemObject(mem);
        }
        if (kernel != nullptr) clReleaseKernel(kernel);
        if (program != nullptr) clReleaseProgram(program);
        if (queue != nullptr) clReleaseCommandQueue(queue);
        if (ctx != nullptr) clReleaseContext(ctx);
    };

    try
    {
        cl_uint nPlatforms = 0;
        check(clGetPlatformIDs(0, nullptr, &nPlatforms), "clGetPlatformIDs(count)");
        if (nPlatforms == 0) throw std::runtime_error("No OpenCL platform found");
        std::vector<cl_platform_id> platforms(nPlatforms);
        check(clGetPlatformIDs(nPlatforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

        cl_device_id device = nullptr;
        for (cl_platform_id platform : platforms)
        {
            cl_uint nDevs = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &nDevs) == CL_SUCCESS && nDevs > 0)
            {
                std::vector<cl_device_id> devices(nDevs);
                check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, nDevs, devices.data(), nullptr), "clGetDeviceIDs(GPU)");
                device = devices[0];
                break;
            }
        }
        if (device == nullptr)
        {
            for (cl_platform_id platform : platforms)
            {
                cl_uint nDevs = 0;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &nDevs) == CL_SUCCESS && nDevs > 0)
                {
                    std::vector<cl_device_id> devices(nDevs);
                    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nDevs, devices.data(), nullptr), "clGetDeviceIDs(ALL)");
                    device = devices[0];
                    break;
                }
            }
        }
        if (device == nullptr) throw std::runtime_error("No OpenCL device found");

        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "clCreateContext");
        queue = clCreateCommandQueue(ctx, device, 0, &err);
        check(err, "clCreateCommandQueue");

        std::string kernelPath = "opencl_kernels.cl";
        const char* kernelPathEnv = std::getenv("CURRENTNE_OPENCL_KERNEL");
        if (kernelPathEnv != nullptr && kernelPathEnv[0] != '\0')
        {
            kernelPath = kernelPathEnv;
        }
        std::string kernelSource = readTextFile(kernelPath);
        if (kernelSource.empty())
        {
            throw std::runtime_error("Cannot load OpenCL kernel source (set CURRENTNE_OPENCL_KERNEL or run where opencl_kernels.cl exists)");
        }

        const char* sourcePtr = kernelSource.c_str();
        size_t sourceLen = kernelSource.size();
        program = clCreateProgramWithSource(ctx, 1, &sourcePtr, &sourceLen, &err);
        check(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t logSize = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::string buildLog(logSize, '\0');
            if (logSize > 1)
            {
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
            }
            throw std::runtime_error("clBuildProgram failed:\n" + buildLog);
        }

        kernel = clCreateKernel(program, "kernel_pairs_tiled", &err);
        check(err, "clCreateKernel");

        const size_t genoBytes = static_cast<size_t>(L) * static_cast<size_t>(N) * sizeof(char);
        cl_mem d_geno = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, genoBytes, (void*)genoT, &err);
        check(err, "clCreateBuffer(geno)");
        toRelease.push_back(d_geno);

        std::vector<char> zeroCromo;
        const char* cromoInput = cromo;
        if (cromoInput == nullptr)
        {
            zeroCromo.assign(static_cast<size_t>(L), 0);
            cromoInput = zeroCromo.data();
        }
        cl_mem d_cromo = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, static_cast<size_t>(L) * sizeof(char), (void*)cromoInput, &err);
        check(err, "clCreateBuffer(cromo)");
        toRelease.push_back(d_cromo);

        std::vector<double> zeroPos;
        const double* posiInput = posiCM;
        if (posiInput == nullptr)
        {
            zeroPos.assign(static_cast<size_t>(L), 0.0);
            posiInput = zeroPos.data();
        }
        cl_mem d_posi = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, static_cast<size_t>(L) * sizeof(double), (void*)posiInput, &err);
        check(err, "clCreateBuffer(posi)");
        toRelease.push_back(d_posi);

        cl_mem d_cp = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_cx = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_xD = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        cl_mem d_xW = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        cl_mem d_cp05 = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_cx05 = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_xD05 = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        cl_mem d_xW05 = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        cl_mem d_cplink = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_cxlink = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(long int));
        cl_mem d_xDlink = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        cl_mem d_xWlink = makeZeroedBuffer(ctx, queue, static_cast<size_t>(L) * sizeof(double));
        toRelease.insert(toRelease.end(), {d_cp, d_cx, d_xD, d_xW, d_cp05, d_cx05, d_xD05, d_xW05, d_cplink, d_cxlink, d_xDlink, d_xWlink});

        unsigned char f_chr = flag_chr ? 1u : 0u;
        unsigned char f_cM = flag_cM ? 1u : 0u;
        int arg = 0;
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_geno), "set arg geno");
        check(clSetKernelArg(kernel, arg++, sizeof(int), &N), "set arg N");
        check(clSetKernelArg(kernel, arg++, sizeof(int), &L), "set arg L");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cromo), "set arg cromo");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_posi), "set arg posi");
        check(clSetKernelArg(kernel, arg++, sizeof(unsigned char), &f_chr), "set arg flag_chr");
        check(clSetKernelArg(kernel, arg++, sizeof(unsigned char), &f_cM), "set arg flag_cM");
        check(clSetKernelArg(kernel, arg++, sizeof(double), &z_cm), "set arg z");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cp), "set arg cp");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cx), "set arg cx");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xD), "set arg xD");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xW), "set arg xW");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cp05), "set arg cp05");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cx05), "set arg cx05");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xD05), "set arg xD05");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xW05), "set arg xW05");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cplink), "set arg cplink");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_cxlink), "set arg cxlink");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xDlink), "set arg xDlink");
        check(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &d_xWlink), "set arg xWlink");

        size_t local = 128;
        size_t maxKernelLocal = 0;
        if (clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxKernelLocal, nullptr) == CL_SUCCESS)
        {
            if (maxKernelLocal > 0) local = std::min(local, maxKernelLocal);
        }
        if (local == 0) local = 1;

        const int maxGroupsPerLaunch = 4096;
        for (int i0 = 0; i0 < (L - 1);)
        {
            const int iCount = std::min(maxGroupsPerLaunch, (L - 1) - i0);
            check(clSetKernelArg(kernel, 20, sizeof(int), &i0), "set arg i0");
            check(clSetKernelArg(kernel, 21, sizeof(int), &iCount), "set arg i_count");
            const size_t global = static_cast<size_t>(iCount) * local;
            check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "enqueue kernel");
            i0 += iCount;
        }
        check(clFinish(queue), "clFinish");

        check(clEnqueueReadBuffer(queue, d_cp, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_contapares, 0, nullptr, nullptr), "read cp");
        check(clEnqueueReadBuffer(queue, d_cx, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_containdX, 0, nullptr, nullptr), "read cx");
        check(clEnqueueReadBuffer(queue, d_xD, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xD, 0, nullptr, nullptr), "read xD");
        check(clEnqueueReadBuffer(queue, d_xW, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xW, 0, nullptr, nullptr), "read xW");
        check(clEnqueueReadBuffer(queue, d_cp05, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_contapares05, 0, nullptr, nullptr), "read cp05");
        check(clEnqueueReadBuffer(queue, d_cx05, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_containdX05, 0, nullptr, nullptr), "read cx05");
        check(clEnqueueReadBuffer(queue, d_xD05, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xD05, 0, nullptr, nullptr), "read xD05");
        check(clEnqueueReadBuffer(queue, d_xW05, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xW05, 0, nullptr, nullptr), "read xW05");
        check(clEnqueueReadBuffer(queue, d_cplink, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_contapareslink, 0, nullptr, nullptr), "read cplink");
        check(clEnqueueReadBuffer(queue, d_cxlink, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(long int), x_containdXlink, 0, nullptr, nullptr), "read cxlink");
        check(clEnqueueReadBuffer(queue, d_xDlink, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xDlink, 0, nullptr, nullptr), "read xDlink");
        check(clEnqueueReadBuffer(queue, d_xWlink, CL_TRUE, 0, static_cast<size_t>(L) * sizeof(double), xWlink, 0, nullptr, nullptr), "read xWlink");
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    cleanup();
}
