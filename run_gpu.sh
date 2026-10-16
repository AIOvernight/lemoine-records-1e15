#!/bin/bash
# The GPU production run of 2026-10-15, as it was started (command lines reconstructed from the START lines of
# log/gpu_3.6e15.log and the run notes). It continues Shimizu's run from N = 2^28 + 931323 * 2^30 = 1000000725188608
# with the same grid (W = 2^30, SEG = 240, K = 128) and the same statistics in the DONE lines.
# The cumulative prime count is continued from pi(N-1) = 29844591414555 (PICUM) so that the DONE lines chain onto
# log/l1e15.log. RECA/RECB are the last records of that log; a new REC line is written only above them.
#
# Three logs were produced (concatenated in order in log/gpu_3.6e15.log):
#   prod1.log  lemgpu  (first version)  [N, 1054890407231488)        213 DONE lines
#   prod2.log  lemgpu3 (optimised)      [1054890407231488, 1059013575835648)  16 DONE lines
#   prod3.log  lemgpu3                  [1059013575835648, 3607647169282048)  9890 DONE lines, stopped at a DONE boundary
# The START line of each log is written by the program; it labels the binary "gpu=lemgpu" in all three.
# END was set to 5e15; the run was stopped by the operator after the DONE line ending at E = 3607647169282048.
# SELFCHECK=k: every k-th window the GPU prime bitmap is compared bit for bit with a primesieve bitmap.
# NAIVE=k: every k-th window a per-n CPU scan recomputes the uncovered counts and maxima.
# Total wall time of the 10119 DONE lines: 22732 s = 6.31 h on one RTX 5090 (mean 1.15e11 n/s).
#
# Usage: bash run_gpu.sh        (writes log/gpu_prod.log; needs ./lemgpu3 from build_gpu.sh)
set -e
N=1000000725188608
END=5000000404717568
PICUM=29844591414555
# prod1 (lemgpu): SELFCHECK every 97 windows, NAIVE every 503 windows
# ./lemgpu  $END START=$N PICUM=$PICUM RECA=13037 RECB=5881 SEG=240 K=128 SUBBITS=19 SELFCHECK=97  NAIVE=503  LOG=log/prod1.log
# prod2, prod3 (lemgpu3): SELFCHECK every 251 windows, NAIVE every 2003 windows
# ./lemgpu3 $END START=1054890407231488 PICUM=31432571651404  RECA=13037 RECB=5881 SEG=240 K=128 SUBBITS=19 SELFCHECK=251 NAIVE=2003 LOG=log/prod2.log
# ./lemgpu3 $END START=1059013575835648 PICUM=31551758497606  RECA=13037 RECB=5881 SEG=240 K=128 SUBBITS=19 SELFCHECK=251 NAIVE=2003 LOG=log/prod3.log
./lemgpu3 "$END" START="$N" PICUM="$PICUM" RECA=13037 RECB=5881 SEG=240 K=128 SUBBITS=19 SELFCHECK=251 NAIVE=2003 LOG=log/gpu_prod.log

# To resume after E = 3607647169282048 (the end of log/gpu_3.6e15.log):
# ./lemgpu3 $END START=3607647169282048 PICUM=103694116073174 RECA=13841 RECB=6337 SEG=240 K=128 SUBBITS=19 SELFCHECK=251 NAIVE=2003 LOG=log/gpu_prod.log
