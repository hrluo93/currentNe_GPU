
#include "gpu_ld.cuh"
#include "lib/progress.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::abort(); \
    } \
} while(0)
#endif

ProgressStatus* g_progress = nullptr;

__device__ inline void atomicAddLL(long long* addr, long long val){
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(val));
}

__global__ void kernel_pairs_tiled(
    const char* __restrict__ genoT, int N, int L,
    const int* __restrict__ cromo, const double* __restrict__ posiCM,
    const bool flag_chr, const bool flag_cM, const double z_cm,
    long long* __restrict__ x_contapares,  long long* __restrict__ x_containdX,  double* __restrict__ xD,  double* __restrict__ xW,
    long long* __restrict__ x_contapares05,long long* __restrict__ x_containdX05,double* __restrict__ xD05,double* __restrict__ xW05,
    long long* __restrict__ x_contapareslink,long long* __restrict__ x_containdXlink,double* __restrict__ xDlink,double* __restrict__ xWlink,
    int i0, int Ltot
){
    int i = i0 + blockIdx.x;
    if (i >= Ltot) return;

    for (int j = i + 1 + threadIdx.x; j < Ltot; j += blockDim.x) {
        int cnt = 0;
        double tacui = 0.0, tacuj = 0.0;
        int tHoHo = 0, tHoHetHetHo = 0, tHetHet = 0;

        const char* gi_base = genoT + (size_t)i * N;
        const char* gj_base = genoT + (size_t)j * N;

        for (int r = 0; r < N; ++r) {
            int gi = (int)gi_base[r];
            int gj = (int)gj_base[r];
            int ss = gi + gj;
            if (ss < 9) {
                tacui += gj;
                tacuj += gi;
                ++cnt;
                if (ss == 2) { if (gi == gj) ++tHetHet; }
                else if (ss == 3) { ++tHoHetHetHo; }
                else if (ss == 4) { ++tHoHo; }
            }
        }

        if (cnt > 0) {
            tacui /= (cnt * 2.0);
            tacuj /= (cnt * 2.0);
            double W = tacui * tacuj;
            double D = -2.0 * W + (2.0 * double(tHoHo) + double(tHoHetHetHo) + 0.5 * double(tHetHet)) / double(cnt);
            D *= D;
            W *= (1.0 - tacui) * (1.0 - tacuj);

            atomicAddLL(&x_contapares[j], 1);
            atomicAddLL(&x_containdX[j], (long long)cnt);
            atomicAdd(&xD[j], D);
            atomicAdd(&xW[j], W);

            if (flag_chr) {
                if (cromo[i] != cromo[j]) {
                    atomicAddLL(&x_contapares05[j], 1);
                    atomicAddLL(&x_containdX05[j], (long long)cnt);
                    atomicAdd(&xD05[j], D);
                    atomicAdd(&xW05[j], W);
                } else {
                    bool takeLink = true;
                    if (flag_cM) {
                        double d = fabs(posiCM[i] - posiCM[j]);
                        takeLink = (d > z_cm);
                    }
                    if (takeLink) {
                        atomicAddLL(&x_contapareslink[j], 1);
                        atomicAddLL(&x_containdXlink[j], (long long)cnt);
                        atomicAdd(&xDlink[j], D);
                        atomicAdd(&xWlink[j], W);
                    }
                }
            }
        }
    }
}

void ComputeLD_GPU(
    const char* genoT, int N, int L,
    const int* cromo, const double* posiCM,
    bool flag_chr, bool flag_cM, double z_cm,
    long long* x_contapares,  long long* x_containdX,  double* xD,  double* xW,
    long long* x_contapares05,long long* x_containdX05,double* xD05,double* xW05,
    long long* x_contapareslink,long long* x_containdXlink,double* xDlink,double* xWlink
){
    char *d_genoT = nullptr;
    int  *d_cromo = nullptr;
    double *d_posi = nullptr;

    long long *d_cp=nullptr,*d_cx=nullptr,*d_cp05=nullptr,*d_cx05=nullptr,*d_cplink=nullptr,*d_cxlink=nullptr;
    double *d_xD=nullptr,*d_xW=nullptr,*d_xD05=nullptr,*d_xW05=nullptr,*d_xDlink=nullptr,*d_xWlink=nullptr;

    size_t geno_bytes = (size_t)L * (size_t)N * sizeof(char);
    CUDA_CHECK(cudaMalloc(&d_genoT, geno_bytes));
    CUDA_CHECK(cudaMemcpy(d_genoT, genoT, geno_bytes, cudaMemcpyHostToDevice));

    if (flag_chr) {
        CUDA_CHECK(cudaMalloc(&d_cromo, L * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_cromo, cromo, L * sizeof(int), cudaMemcpyHostToDevice));
        if (flag_cM) {
            CUDA_CHECK(cudaMalloc(&d_posi, L * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_posi, posiCM, L * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    auto zmem = [](void* p, size_t b){ CUDA_CHECK(cudaMemset(p, 0, b)); };

    CUDA_CHECK(cudaMalloc(&d_cp, L*sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_cx, L*sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_xD, L*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xW, L*sizeof(double)));
    zmem(d_cp, L*sizeof(long long));
    zmem(d_cx, L*sizeof(long long));
    zmem(d_xD, L*sizeof(double));
    zmem(d_xW, L*sizeof(double));

    if (flag_chr){
        CUDA_CHECK(cudaMalloc(&d_cp05, L*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_cx05, L*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_xD05, L*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_xW05, L*sizeof(double)));
        zmem(d_cp05, L*sizeof(long long));
        zmem(d_cx05, L*sizeof(long long));
        zmem(d_xD05, L*sizeof(double));
        zmem(d_xW05, L*sizeof(double));

        CUDA_CHECK(cudaMalloc(&d_cplink, L*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_cxlink, L*sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_xDlink, L*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_xWlink, L*sizeof(double)));
        zmem(d_cplink, L*sizeof(long long));
        zmem(d_cxlink, L*sizeof(long long));
        zmem(d_xDlink, L*sizeof(double));
        zmem(d_xWlink, L*sizeof(double));
    }

    const int CHUNK = 1024;
    for (int i0 = 0; i0 < L; i0 += CHUNK) {
        int nThis = (i0 + CHUNK <= L) ? CHUNK : (L - i0);
        dim3 grid(nThis), block(256);
        kernel_pairs_tiled<<<grid, block>>>(
            d_genoT, N, L,
            d_cromo, d_posi,
            flag_chr, flag_cM, z_cm,
            d_cp, d_cx, d_xD, d_xW,
            d_cp05, d_cx05, d_xD05, d_xW05,
            d_cplink, d_cxlink, d_xDlink, d_xWlink,
            i0, L
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (g_progress) g_progress->SetTaskProgress(i0 + nThis);
    }

    CUDA_CHECK(cudaMemcpy(x_contapares, d_cp, L*sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_containdX, d_cx, L*sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xD,          d_xD, L*sizeof(double),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xW,          d_xW, L*sizeof(double),    cudaMemcpyDeviceToHost));

    if (flag_chr){
        CUDA_CHECK(cudaMemcpy(x_contapares05, d_cp05, L*sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x_containdX05,  d_cx05, L*sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xD05,           d_xD05, L*sizeof(double),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xW05,           d_xW05, L*sizeof(double),    cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(x_contapareslink, d_cplink, L*sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x_containdXlink,  d_cxlink, L*sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xDlink,           d_xDlink, L*sizeof(double),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xWlink,           d_xWlink, L*sizeof(double),    cudaMemcpyDeviceToHost));
    }

    cudaFree(d_genoT);
    if (d_cromo) cudaFree(d_cromo);
    if (d_posi)  cudaFree(d_posi);
    cudaFree(d_cp); cudaFree(d_cx); cudaFree(d_xD); cudaFree(d_xW);
    if (flag_chr){
        cudaFree(d_cp05); cudaFree(d_cx05); cudaFree(d_xD05); cudaFree(d_xW05);
        cudaFree(d_cplink); cudaFree(d_cxlink); cudaFree(d_xDlink); cudaFree(d_xWlink);
    }
}
