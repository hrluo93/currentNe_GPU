
#pragma once
#include <cstdint>

class ProgressStatus;
extern ProgressStatus* g_progress;

void ComputeLD_GPU(
    const char* genoT, int N, int L,
    const int* cromo, const double* posiCM,
    bool flag_chr, bool flag_cM, double z_cm,
    long long* x_contapares,  long long* x_containdX,  double* xD,  double* xW,
    long long* x_contapares05,long long* x_containdX05,double* xD05,double* xW05,
    long long* x_contapareslink,long long* x_containdXlink,double* xDlink,double* xWlink
);
