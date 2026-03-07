#pragma once

void ComputeLD_OpenCL(
    const char* genoT, int N, int L,
    const char* cromo, const double* posiCM,
    bool flag_chr, bool flag_cM, double z_cm,
    long int* x_contapares, long int* x_containdX, double* xD, double* xW,
    long int* x_contapares05, long int* x_containdX05, double* xD05, double* xW05,
    long int* x_contapareslink, long int* x_containdXlink, double* xDlink, double* xWlink);
