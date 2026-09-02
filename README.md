# currentNe_GPU

 Modified GPU-accelerated `currentNe`(https://github.com/esrud/currentNe) that delivers a more than **20-fold** speedup and **only 1-5Gb free GPU memory required**, while producing results **are fully consistent with the original CPU implementation**, including current effective population size (Nₑ) estimates and confidence intervals.
 
The GPU path computes weighted LD (d²) in **FP64** using atomicAdd(double*), while Nₑ and CIs follow the original integration and neural-network variance model.

Requires: **NVIDIA GPU ≥ Pascal (SM ≥ 6.0)**, NVIDIA driver + CUDA Toolkit (12+), gcc/g++ & make, **and ~1 GB free GPU memory** (more for large datasets).

**OpenCL version is available** Under **Release  currentNe-ocl.zip** and can be used on Nvidia, AMD, and Intel GPUs.
**The CUDA and OpenCL implementations produce results that are fully consistent with the original CPU version.**

An Apple Metal FP32 version is also available for testing purposes. Because **Metal does not support FP64, the FP32 estimated d² and Ne values may differ from those of the CPU version**. The FP32 version is provided only as a test of Metal GPU computing (individuals x SNPs pairs=132416165908, M3pro CPU 464.501 sec **Vs.** GPU 0.448286 sec). If needed, please contact the CurrentNe_gpu author: hrluo93@foxmail.com .

## Benchmark
<img width="1942" height="1118" alt="2adc8596-91cf-45c7-a114-bd76c468ed27" src="https://github.com/user-attachments/assets/ed0bb2f3-2682-47c7-9c66-7d370b02cacf" />

**Cooling note:** **Not recommended to run on passively cooled (fanless) Tesla GPUs** without server-grade, front-to-back airflow. The FP64 path saturates the FP units for extended periods, creating stress-test-level thermal load (stress FPU). Inadequate airflow will cause throttling or faults.

CurrentNe original Authors: Enrique Santiago, Carlos Köpke

Citations:

Santiago, E., Caballero, A., Köpke, C., & Novo, I. (2024). Estimation of the contemporary effective population size from SNP data while accounting for mating structure. Molecular Ecology Resources, 24, e13890. https://doi.org/10.1111/1755-0998.13890

Santiago, E., Köpke, C. & Caballero, A. Accounting for population structure and data quality in demographic inference with linkage disequilibrium methods. Nat Commun 16, 6054 (2025). https://doi.org/10.1038/s41467-025-61378-w## 


### CUDA build (recommended)
```bash
unzip currentNe_gpu_full.zip
cd currentNe_gpu_full
make ARCH=sm_89        # choose your GPU's SM arch (sm_70, sm_80, sm_86, sm_89 ...) also should be set `ARCH ?=sm_89` in Makefile accordingly.
```
This creates `./currentNe_gpu`.

### CPU fallback
```bash
make cpu
```
This creates `./currentNe_gpu_cpu` (OpenMP).

### OpenCL build (Tested on Nvidia GPU)
```bash
unzip currentNe-ocl.zip
cd currentNe-ocl
make -f Makefile.opencl
```
This creates `./currentNe_ocl`.

## Run

General form:
```bash
ulimit -s unlimited    #default Maxloci setting to 20 million, can increase in the cpp file.
./currentNe_gpu <datafile> <num_chromosomes> [options]
```

- `<datafile>`: one of
  - `prefix.vcf`
  - `prefix.ped` (requires `prefix.map` in the same folder)
  - `prefix.tped` (with individuals as columns following the first 4 fields)
- `<num_chromosomes>`: required (e.g., `22` for human-like autosomes, or the true count for your organism's autosomes).

Common options:
- `-s <N>`  Number of SNPs to use (default: all segregating)  
- `-t <T>`  CPU threads (for non-GPU parts; default: OpenMP auto)  
- `-o <file>` Output filename (default: `<prefix>_currentNe_OUTPUT.txt`)  
- `-k <int>` Important, please see original description in currentNe 
- `-q`      Quiet: only print Ne (and with `-v` also 50% & 90% CI)  
- `-v`      With `-q`, also print CIs  
- `-p`      Print full analysis to stdout instead of file

Examples:
```bash
# TPED
./currentNe_gpu mydata.tped 19 -t 8

# PED/MAP
./currentNe_gpu mypop.ped 19 -t 8

# VCF
./currentNe_gpu cohort.vcf 19 -t 8
./currentNe_gpu cohort.vcf 19 -t 8 -k 1 
```
**-t 8 is enough**

## Output
- Full report file (unless `-p`): `<prefix>_currentNe_OUTPUT.txt`  
  Includes: input stats, d², expected/observed het, **Ne point estimate**, **50%/90% CI**.

## Notes
- Double `atomicAdd` requires GPU architecture **sm_60+**; set `ARCH` accordingly.
- For very large SNP counts, memory = `L × N` bytes (char). Consider filtering `-s` or thinning SNPs.
