#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline void atomic_add_long(__global long* addr, long val)
{
    volatile __global ulong* uaddr = (volatile __global ulong*)addr;
    ulong old_bits = *uaddr;
    while (1)
    {
        long old_val = (long)old_bits;
        long new_val = old_val + val;
        ulong prev = atomic_cmpxchg(uaddr, old_bits, (ulong)new_val);
        if (prev == old_bits) break;
        old_bits = prev;
    }
}

inline void atomic_add_double(__global double* addr, double val)
{
    volatile __global ulong* uaddr = (volatile __global ulong*)addr;
    ulong old_bits = *uaddr;
    while (1)
    {
        double old_val = as_double(old_bits);
        double new_val = old_val + val;
        ulong prev = atomic_cmpxchg(uaddr, old_bits, as_ulong(new_val));
        if (prev == old_bits) break;
        old_bits = prev;
    }
}

__kernel void kernel_pairs_tiled(
    __global const char* genoT,
    const int N,
    const int L,
    __global const char* cromo,
    __global const double* posiCM,
    const char flag_chr,
    const char flag_cM,
    const double z_cm,
    __global long* x_contapares,
    __global long* x_containdX,
    __global double* xD,
    __global double* xW,
    __global long* x_contapares05,
    __global long* x_containdX05,
    __global double* xD05,
    __global double* xW05,
    __global long* x_contapareslink,
    __global long* x_containdXlink,
    __global double* xDlink,
    __global double* xWlink,
    const int i0,
    const int i_count)
{
    const int group = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    if (group >= i_count) return;
    const int i = i0 + group;
    if (i >= (L - 1)) return;

    __global const char* gi_base = genoT + ((size_t)i * (size_t)N);

    for (int j = i + 1 + lid; j < L; j += lsize)
    {
        int cnt = 0;
        double tacui = 0.0;
        double tacuj = 0.0;
        int tHoHo = 0;
        int tHoHetHetHo = 0;
        int tHetHet = 0;
        __global const char* gj_base = genoT + ((size_t)j * (size_t)N);

        for (int r = 0; r < N; ++r)
        {
            const int gi = (int)gi_base[r];
            const int gj = (int)gj_base[r];
            const int ss = gi + gj;
            if (ss < 9)
            {
                tacui += (double)gj;
                tacuj += (double)gi;
                ++cnt;
                if (ss == 2)
                {
                    if (gi == gj) ++tHetHet;
                }
                else if (ss == 3)
                {
                    ++tHoHetHetHo;
                }
                else if (ss == 4)
                {
                    ++tHoHo;
                }
            }
        }

        if (cnt > 0)
        {
            tacui /= ((double)cnt * 2.0);
            tacuj /= ((double)cnt * 2.0);
            double W = tacui * tacuj;
            double D = -2.0 * W + (2.0 * (double)tHoHo + (double)tHoHetHetHo + ((double)tHetHet / 2.0)) / (double)cnt;
            D *= D;
            W *= (1.0 - tacui) * (1.0 - tacuj);

            atomic_add_long(&x_contapares[j], 1);
            atomic_add_long(&x_containdX[j], (long)cnt);
            atomic_add_double(&xD[j], D);
            atomic_add_double(&xW[j], W);

            if (flag_chr)
            {
                if (cromo[i] != cromo[j])
                {
                    atomic_add_long(&x_contapares05[j], 1);
                    atomic_add_long(&x_containdX05[j], (long)cnt);
                    atomic_add_double(&xD05[j], D);
                    atomic_add_double(&xW05[j], W);
                }
                else
                {
                    char keep_link = 1;
                    if (flag_cM)
                    {
                        const double distance = fabs(posiCM[j] - posiCM[i]);
                        keep_link = (distance > z_cm) ? 1 : 0;
                    }
                    if (keep_link)
                    {
                        atomic_add_long(&x_contapareslink[j], 1);
                        atomic_add_long(&x_containdXlink[j], (long)cnt);
                        atomic_add_double(&xDlink[j], D);
                        atomic_add_double(&xWlink[j], W);
                    }
                }
            }
        }
    }
}
