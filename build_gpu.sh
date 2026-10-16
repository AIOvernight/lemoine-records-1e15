#!/bin/bash
# Build the GPU verifier (src/lemgpu3.cu), its first version (src/lemgpu.cu) and the independent CPU
# recomputation (src/cpucheck.c). Requirements: CUDA 12.x (nvcc), an NVIDIA GPU (the production run used an
# RTX 5090, sm_120; set ARCH for another GPU), primesieve (headers + library), gcc with OpenMP.
# Usage: bash build_gpu.sh [ARCH=sm_120] [PRIMESIEVE_PREFIX=/usr/local]
# Prints the first 12 hex digits of each source md5.
set -e
ARCH=${1:-sm_120}
P=${2:-/usr/local}
nvcc -O3 -arch="$ARCH" -Xcompiler -fopenmp -I"$P/include" -L"$P/lib" src/lemgpu3.cu -o lemgpu3 -lprimesieve
nvcc -O3 -arch="$ARCH" -Xcompiler -fopenmp -I"$P/include" -L"$P/lib" src/lemgpu.cu -o lemgpu -lprimesieve
gcc -O3 -march=native -fopenmp -Wall -I"$P/include" -L"$P/lib" src/cpucheck.c -o cpucheck -lprimesieve
for s in src/lemgpu3.cu src/lemgpu.cu src/cpucheck.c; do echo "built $(basename "$s" | sed 's/\..*//') src=$(md5sum "$s" | cut -c1-12)"; done
