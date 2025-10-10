#include "gpu_ld.cuh"
#include <vector>
#ifdef USE_CUDA_NE
#include <cuda_runtime.h>
ProgressStatus* g_progress = nullptr;
static bool cuda_available() { int n=0; return (cudaGetDeviceCount(&n)==cudaSuccess && n>0); }
#endif
params.progress.InitCurrentTask(eneloc-1, "Measuring d^2");
params.progress.SetTaskProgress(0);
params.progress.SaveProgress();
const int Lsel = eneloc;
const int Nsel = eneind;
std::vector<char> genoT((size_t)Lsel*(size_t)Nsel);
for (int jj=0; jj<Lsel; ++jj) {
    int loc = valid_idx[jj];
    for (int ii=0; ii<Nsel; ++ii) {
        int ind = pind[ii];
        genoT[(size_t)jj*Nsel + ii] = indi[ind][loc];
    }
}
std::vector<int>  cromo_sel(Lsel);
std::vector<double> posiCM_sel(Lsel);
if (flag_chr) {
    for (int jj=0; jj<Lsel; ++jj) {
        int loc = valid_idx[jj];
        cromo_sel[jj] = cromo[loc];
        posiCM_sel[jj] = posiCM[loc];
    }
}
std::vector<long long> x_contapares_v(Lsel,0), x_containdX_v(Lsel,0);
std::vector<double>    xD_v(Lsel,0.0), xW_v(Lsel,0.0);
std::vector<long long> x_contapares05_v(Lsel,0), x_containdX05_v(Lsel,0);
std::vector<double>    xD05_v(Lsel,0.0), xW05_v(Lsel,0.0);
std::vector<long long> x_contapareslink_v(Lsel,0), x_containdXlink_v(Lsel,0);
std::vector<double>    xDlink_v(Lsel,0.0), xWlink_v(Lsel,0.0);
bool usedGPU = false;
#ifdef USE_CUDA_NE
if (cuda_available()) {
    g_progress = &params.progress;
    ComputeLD_GPU(
        genoT.data(), Nsel, Lsel,
        flag_chr ? cromo_sel.data() : nullptr,
        (flag_chr && flag_cM) ? posiCM_sel.data() : nullptr,
        flag_chr, flag_cM, params.z,
        x_contapares_v.data(), x_containdX_v.data(), xD_v.data(), xW_v.data(),
        x_contapares05_v.data(), x_containdX05_v.data(), xD05_v.data(), xW05_v.data(),
        x_contapareslink_v.data(), x_containdXlink_v.data(), xDlink_v.data(), xWlink_v.data()
    );
    usedGPU = true;
}
#endif
if (!usedGPU) {
    for (int j2=0;j2<Lsel-1;++j2){
        int pj_loc = valid_idx[j2];
        #pragma omp parallel for
        for (int j3=j2+1;j3<Lsel;++j3){
            int pi_loc = valid_idx[j3];
            double tacui=0.0, tacuj=0.0;
            int tHoHo=0, tHoHetHetHo=0, tHetHet=0, _containdX=0;
            for (int ii=0; ii<Nsel; ++ii){
                int ind = pind[ii];
                int gi = indi[ind][pi_loc];
                int gj = indi[ind][pj_loc];
                int ss=gi+gj;
                if (ss<9){
                    tacui += gi;
                    tacuj += gj;
                    ++_containdX;
                    if (ss==2){ if (gi==gj) ++tHetHet; }
                    else if (ss==3){ ++tHoHetHetHo; }
                    else if (ss==4){ ++tHoHo; }
                }
            }
            if (_containdX>0){
                tacui/=(2.0*_containdX);
                tacuj/=(2.0*_containdX);
                double W=tacui*tacuj;
                double D=-2.0*W + (2.0*double(tHoHo) + double(tHoHetHetHo) + 0.5*double(tHetHet))/double(_containdX);
                D*=D;
                W*=(1.0-tacui)*(1.0-tacuj);
                #pragma omp atomic
                x_contapares_v[j3] += 1;
                #pragma omp atomic
                x_containdX_v[j3] += _containdX;
                #pragma omp atomic
                xD_v[j3] += D;
                #pragma omp atomic
                xW_v[j3] += W;
                if (flag_chr){
                    if (cromo[pi_loc] != cromo[pj_loc]){
                        #pragma omp atomic
                        x_contapares05_v[j3] += 1;
                        #pragma omp atomic
                        x_containdX05_v[j3] += _containdX;
                        #pragma omp atomic
                        xD05_v[j3] += D;
                        #pragma omp atomic
                        xW05_v[j3] += W;
                    } else {
                        bool takeLink = true;
                        if (flag_cM){
                            double d = fabs(posiCM[pi_loc] - posiCM[pj_loc]);
                            takeLink = (d > params.z);
                        }
                        if (takeLink){
                            #pragma omp atomic
                            x_contapareslink_v[j3] += 1;
                            #pragma omp atomic
                            x_containdXlink_v[j3] += _containdX;
                            #pragma omp atomic
                            xDlink_v[j3] += D;
                            #pragma omp atomic
                            xWlink_v[j3] += W;
                        }
                    }
                }
            }
        }
        if (j2 % 1000 == 0) { params.progress.SetTaskProgress(j2+1); }
    }
}
acun = effndata = acuD2 = acuW = 0.0;
acun05 = effndata05 = acuD205 = acuW05 = 0.0;
acunlink = effndatalink = acuD2link = acuWlink = 0.0;
if (flag_chr) {
    for (int j3=0;j3<Lsel;++j3){
        acun     += (double)x_contapares_v[j3];
        effndata += (double)x_containdX_v[j3];
        acuD2    += xD_v[j3];
        acuW     += xW_v[j3];
        acun05       += (double)x_contapares05_v[j3];
        effndata05   += (double)x_containdX05_v[j3];
        acuD205      += xD05_v[j3];
        acuW05       += xW05_v[j3];
        acunlink     += (double)x_contapareslink_v[j3];
        effndatalink += (double)x_containdXlink_v[j3];
        acuD2link    += xDlink_v[j3];
        acuWlink     += xWlink_v[j3];
    }
    acun = acun05 + acunlink;
    effndata = effndata05 + effndatalink;
    d2s = (acuD205 + acuD2link) / (acuW05 + acuWlink);
    acuD2 = (acuD205 + acuD2link) / acun;
    acuW  = (acuW05  + acuWlink)  / acun;
    d2s05   = (acuW05>0.0)   ? (acuD205/acuW05)     : 0.0;
    acuD205 /= (acun05>0.0   ? acun05   : 1.0);
    acuW05  /= (acun05>0.0   ? acun05   : 1.0);
    d2slink = (acuWlink>0.0) ? (acuD2link/acuWlink) : 0.0;
    acuD2link/= (acunlink>0.0? acunlink : 1.0);
    acuWlink /= (acunlink>0.0? acunlink : 1.0);
} else {
    for (int j3=0;j3<Lsel;++j3){
        acun     += (double)x_contapares_v[j3];
        effndata += (double)x_containdX_v[j3];
        acuD2    += xD_v[j3];
        acuW     += xW_v[j3];
    }
    d2s  = (acuW>0.0) ? (acuD2/acuW) : 0.0;
    acuD2 /= (acun>0.0 ? acun : 1.0);
    acuW  /= (acun>0.0 ? acun : 1.0);
}
