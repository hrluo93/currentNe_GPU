# currentNe_GPU

 Modified GPU-accelerated `currentNe`(https://github.com/esrud/currentNe) with **PED/MAP**, and **VCF** input support, plus **complete Ne estimation & confidence intervals**. 
The GPU computes weighted LD (d²) accumulations; Ne and CIs follow the original integration + neural-net variance model.

Requires: **NVIDIA GPU ≥ Pascal (SM ≥ 6.0)**, NVIDIA driver + CUDA Toolkit (11.4+), gcc/g++ & make, **and ~1 GB free GPU memory** (more for large datasets).

Santiago, E., Caballero, A., Köpke, C., & Novo, I. (2024). Estimation of the contemporary effective population size from SNP data while accounting for mating structure. Molecular Ecology Resources, 24, e13890. https://doi.org/10.1111/1755-0998.13890

Santiago, E., Köpke, C. & Caballero, A. Accounting for population structure and data quality in demographic inference with linkage disequilibrium methods. Nat Commun 16, 6054 (2025). https://doi.org/10.1038/s41467-025-61378-w## 


### CUDA build (recommended)
```bash
unzip currentNe_gpu_full.zip
cd currentNe_gpu_full
make ARCH=sm_89        # choose your GPU's SM arch (sm_70, sm_80, sm_86, sm_89 ...)
ulimit -s unlimited    #default Maxloci setting to 20 million
```
This creates `./currentNe_gpu`.

### CPU fallback
```bash
make cpu
```
This creates `./currentNe_gpu_cpu` (OpenMP).

## Run

General form:
```bash
./currentNe_gpu <datafile> <num_chromosomes> [options]
```

- `<datafile>`: one of
  - `prefix.vcf`
  - `prefix.ped` (requires `prefix.map` in the same folder)
  - `prefix.tped` (with individuals as columns following first 4 fields)
- `<num_chromosomes>`: required (e.g., `23` for human-like, or the true count for your organism).

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
  Includes: input stats, d², expected/observed het, **Ne point estimate**, **50%/90% CI**; plus between-chromosome-only Ne if map/chrom info present.

## Notes
- Double `atomicAdd` requires GPU architecture **sm_60+**; set `ARCH` accordingly.
- For very large SNP counts, memory = `L × N` bytes (char). Consider filtering `-s` or thinning SNPs.
