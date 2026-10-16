# Lemoine's conjecture: the two record sequences extended to 10^15

Prepared 2026/10/13 by Hirotaka Shimizu with the AI assistant Claude (see How this was made).
Status: computational result. No human expert has checked it yet.

**GPU extension (branch `gpu-3.6e15`, 2026-10-15):** the same two sequences were continued on one NVIDIA RTX 5090 from
N to 3.6·10^15 (B) and 7.2·10^15 (A), adding four candidate terms, with purpose-written CUDA kernels (`src/lemgpu3.cu`: wheel
sieve, tiled 128-shift cover, exact-phase scan; 1.15·10^11 numbers per second). See the section "GPU extension" below,
[docs/GPU_KERNELS.md](docs/GPU_KERNELS.md) and [docs/GPU_PROVENANCE.md](docs/GPU_PROVENANCE.md). Everything above that section is Hirotaka Shimizu's original text.

## Claim

Lemoine's conjecture (also called Levy's conjecture) states that every odd n > 5 can be written as n = 2p + q
with p and q prime. Let N = 2^28 + 931323 · 2^30 = 1,000,000,725,188,608.

1. Every odd n with 5 < n < 2N has a representation n = 2p + q. (This follows from the A part below.)
2. **B (least p).** For odd n > 5 let p(n) be the least prime p such that n − 2p is prime.
   The records of p(n) for n < N are the 54 lines `REC B` in `log/l1e15.log`.
   The first 41 are the known terms of OEIS A195353 / A195354; the next 13 are new.
3. **A (least q).** For odd n > 5 let q(n) be the least prime q such that (n − q)/2 is prime.
   The records of q(n) for n < 2N are the 57 lines `REC A` in `log/l1e15.log`.
   The first 50 are the known terms of OEIS A194828 / A194829; the next 7 are new.

A record is an n whose value is larger than the value at every smaller odd n > 5.

## New terms

A (least q), continuing A194828 (n) and A194829 (q). The OEIS tables end at n = 14480831322263, q = 9157
("Checked up to 2*10^13").

| n | q |
|---:|---:|
| 47338506360191 | 10957 |
| 215429154578783 | 11257 |
| 615915775151587 | 11321 |
| 655010575058773 | 11927 |
| 861635479538627 | 11953 |
| 1002833079463907 | 12289 |
| 1350940030581751 | 13037 |

B (least p), continuing A195353 (n) and A195354 (p). The OEIS tables end at n = 8570124441113, p = 3767
("Checked up to n = 10^13").

| n | p |
|---:|---:|
| 10154135153999 | 3851 |
| 13982148897511 | 3907 |
| 18115451658709 | 4093 |
| 19909053765571 | 4261 |
| 20633922827279 | 4493 |
| 51662506459999 | 4549 |
| 70078798991239 | 4813 |
| 82707580488391 | 5281 |
| 388698297570001 | 5437 |
| 648902959501411 | 5479 |
| 709689847014391 | 5569 |
| 710860716846877 | 5623 |
| 733975808707159 | 5881 |

So q(n) ≤ 13037 for all odd 5 < n < 2N, and p(n) ≤ 5881 for all odd 5 < n < N.

## GPU extension to 3.6·10^15 (B) / 7.2·10^15 (A)

Added 2026-10-15 by an AI agent coordinated by [@AIOvernight](https://x.com/AIOvernight) (see
[docs/GPU_PROVENANCE.md](docs/GPU_PROVENANCE.md)). Status: computational result, no human expert has checked it.
It continues the run above from its end point N on the same grid, with the same conventions, and is calibrated
against `log/l1e15.log`.

### Claim

Let E = 3607647169282048 = N + 10119 · 240 · 2^30. The GPU run checked every odd n with N ≤ n < E for B and
2N ≤ n < 2E for A (`log/gpu_3.6e15.log`, 10119 `DONE` lines, anom = 0). Together with the run above:

1. Every odd n with 5 < n < 2E = 7215294338564096 has a representation n = 2p + q.
2. **B (least p).** For odd n < E the records of p(n) are the 54 `REC B` lines of `log/l1e15.log` followed by the
   3 `REC B` lines of `log/gpu_3.6e15.log`.
3. **A (least q).** For odd n < 2E the records of q(n) are the 57 `REC A` lines of `log/l1e15.log` followed by the
   1 `REC A` line of `log/gpu_3.6e15.log`.

### Candidate new terms

A (least q), continuing A194828 (n) / A194829 (q) after n = 1350940030581751, q = 13037:

| n | q |
|---:|---:|
| 2313425910783103 | 13841 |

B (least p), continuing A195353 (n) / A195354 (p) after n = 733975808707159, p = 5881:

| n | p |
|---:|---:|
| 1880662361770607 | 5939 |
| 2423130987173003 | 6317 |
| 3058699595197417 | 6337 |

So q(n) ≤ 13841 for all odd 5 < n < 2E, and p(n) ≤ 6337 for all odd 5 < n < E. The four terms are listed in
`oeis/candidates_gpu.txt`; they have not been submitted to the OEIS yet.

### Method and CUDA kernels (`src/lemgpu3.cu`)

The whole fast phase runs on the GPU, window by window on Shimizu's grid (X = 2^28 + k · 2^30, W = 2^30,
margin 16384 odd numbers, K = 128, q = 1 excluded, A split by q mod 4):

1. a segmented sieve of Eratosthenes writes the bit array P of the odd numbers in [X − 2·16384 + 1, X + W)
   (shared-memory blocks of 2^19 bits with a wheel for the primes 3..31 for primes below 2^20, global atomic
   updates with mod-30 skipping for the larger primes up to √(X + W));
2. a cover kernel ORs P shifted by the first K small primes of each instance (B; A with q ≡ 1 and q ≡ 3 mod 4)
   in shared-memory tiles grouped by word offset;
3. a two-phase scan of the uncovered n (about 0.22 %) finds the least prime by trying the following small primes
   and emits record candidates; the window chains are merged in order on the host, exactly as in `lemoine.c`.

An n with no solution among the small primes of the margin would be reported as ANOMALY; none occurred.

The six kernels, per window of 2^30 integers (one prime bitmap serves B and both halves of A):

| Kernel | What it does | Launch |
|---|---|---|
| `k_small` | Segmented sieve for the primes 37 … 2^20 in a 64 KB shared-memory composite mask per block; the primes 3 … 31 are applied as a wheel of precomputed residue masks instead of being sieved | 1025 blocks × 1024 threads |
| `k_large` | One thread per sieving prime from 2^20 to √(X + W), `atomicAnd` on the global bitmap, skipping the multiples whose cofactor is divisible by 3 or 5 (mod-30 gap table) | one thread per prime, 256 per block |
| `k_count` | Popcount of the primes in the window (warp shuffle, one atomic per warp); the `piseg`/`pi` fields checked against primecount | 1024 × 256, grid-stride |
| `k_cover` | **The fast phase.** Each block copies a tile of the bitmap into shared memory and ORs it shifted by the K = 128 small primes of the instance, the shifts grouped by word offset so that each output word needs a handful of shared loads; a thread owns 32 odd n and emits the uncovered ones (about 0.22 %) to a list | one thread per 32 n, 256 per block |
| `k_scan` | **The exact phase.** One thread per uncovered n tries the following small primes in ascending order until the partner is prime; the least one is the value of the record function at n, compared with the running record, candidates above it are emitted | 4096 × 256 over the uncovered list |
| `k_fin` | Closes the window's statistics and flags an overflow of the uncovered list as an anomaly (never happened) | 1 thread |

Records are chained on the host in order of n, exactly as in `lemoine.c`. A kernel-by-kernel description (memory layout,
the wheel, the tiled cover, the exact phase, throughput, and the differences from the first version `src/lemgpu.cu`) is
in [docs/GPU_KERNELS.md](docs/GPU_KERNELS.md). The CPU recomputation `src/cpucheck.c` shares no code with the kernels.
`src/lemgpu.cu` is the first version (no wheel, no tiles); it produced the first 213 `DONE` lines. The optimised
version did the rest at 1.3–1.5·10^11 numbers per second; the mean over the run, including checks, was
1.15·10^11 numbers per second, 6.31 hours of wall time on one RTX 5090 for 2.6·10^15 numbers (the CPU run above:
71.5 hours on 12 threads for 10^15). Options: `SELFCHECK=k` compares the GPU bit array of every k-th window bit for bit
with a primesieve bit array; `NAIVE=k` recomputes the uncovered counts and maxima of every k-th window on the CPU
by a direct per-n scan; `PICUM=` continues the running prime count from the preceding log.

### Verification

| Check | What it shows | Script / file | Output |
|---|---|---|---|
| Calibration | on the first and the last segment of Shimizu's grid, both GPU programs give the same `DONE` fields and the same 17 `REC` lines as `log/l1e15.log` | `verify/gpu_calibration.md` | `verify/gpu_calibration/` |
| Exhaustive run | every n in [N, E) (B) and [2N, 2E) (A), one program | `src/lemgpu3.cu` | `log/gpu_3.6e15.log` |
| Coverage of the log | the 10119 `DONE` lines run from N to E without gap or overlap, widths 240·2^30, `pi` cumulative from π(N − 1), anom = 0, every `SELFCHECK` and `NAIVE` line ends in `ok`, the 4 `REC` lines are records above 13037 / 5881 | `verify/gpu_continuity.py` | `verify/gpu_continuity.txt` |
| Coverage with Shimizu's script | his `verify/continuity.py` on the chain `log/l1e15.log` + `log/gpu_3.6e15.log`: all 14000 `DONE` lines chain from 2^28 to E and `pi` is cumulative; the remaining 6 issues it reports are header conventions only (4 `START` lines, no `ALLDONE`/`EXIT`, his last segment of 123·2^30 now in the middle) | `verify/continuity.py` | `verify/gpu_continuity.txt` |
| Sieve self-check | 10031 windows (every 97th / 251st): GPU prime bit array equal to primesieve's, `words_diff=0` | `SELFCHECK` lines | `log/gpu_3.6e15.log` |
| Per-n recheck | 1296 windows (every 503rd / 2003rd): uncovered counts and maxima recomputed on the CPU per n, all `ok` | `NAIVE` lines | `log/gpu_3.6e15.log` |
| π(E) | the cumulative prime count of the last `DONE` line, 103694116073174, equals primecount's π(E − 1); the sum of `piseg` equals π(E − 1) − π(N − 1) | primecount | `verify/gpu_pi_check.txt` |
| Value at each record | the 4 `REC` lines recomputed from the definitions with a deterministic Miller–Rabin test, and the maximum over the 100000 preceding odd n is smaller; also Shimizu's `verify/naive_check.py`: 4/4 match | `verify/gpu_naive_check.py`, `verify/naive_check.py` | `verify/gpu_naive_check.txt` |
| Segment recomputation | 6 full segments (the first, segment 1500, and the 4 that contain the records) recomputed by `src/cpucheck.c`, an independent CPU program (primesieve bit array, direct per-n scan, no shared code with the kernels): `piseg`, `uncB`, `uncA`, `maxB`, `maxA` identical, and each record line reproduced | `src/cpucheck.c` | `verify/gpu_cpucheck_summary.txt`, `verify/gpu_cpucheck/` |

Limits: as for the CPU run, the record property over the whole range rests on the one exhaustive GPU pass; the
independent recomputation covers 6 of the 10119 segments (0.06 %) in full, plus the sampled windows above. A second
full pass with a different program would make it airtight. The run was stopped by the operator at a `DONE` boundary
(END had been set to 5·10^15), so the log has no `ALLDONE` line; E is the `E` of its last `DONE` line.

### Reproduce

Requirements: CUDA 12.x with an NVIDIA GPU (the code targets sm_120; pass another `ARCH` to `build_gpu.sh`),
primesieve (headers and library), gcc with OpenMP, Python 3, primecount for the π check.

```bash
bash build_gpu.sh sm_120 /usr/local          # lemgpu3, lemgpu, cpucheck
python3 verify/gpu_continuity.py log/gpu_3.6e15.log
python3 verify/gpu_naive_check.py log/gpu_3.6e15.log 100000
python3 verify/naive_check.py log/gpu_3.6e15.log
primecount 3607647169282047                   # expect 103694116073174
./cpucheck 3058492650815488 3058750348853248 13037 6317   # one record segment, ~5 min on 16 threads; compare with its DONE line
bash run_gpu.sh                               # the full continuation (hours); see the file for the resume command
```

## Prior work

- OEIS A194828 / A194829 and A195353 / A195354 (Hugo Pfoertner, 2011): b-files with 50 and 41 terms, checked up to
  2·10^13 and 10^13. These were still the latest tables when the OEIS pages were read on 2026/10/13.
- Z. Juhász, M. Bartalos, P. Magyar, G. Farkas, *Empirical verification of a new generalisation of Goldbach's
  conjecture up to 10^12 (or 10^13) for all coefficients ≤ 40*, arXiv:2304.00024 (2023). The abstract states
  that Lemoine's conjecture is verified up to 10^13.
- D. Ndegwa, L. Njagi, S. Luketero, B. Nzimbi, K. Benjamin, *Extending Computational Verification of Lemoine's
  Conjecture to 1500 Digit Odd Numbers*, International Journal of Mathematics Trends and Technology 72(3) (2026).
  This tests randomly chosen odd numbers of up to 1500 digits with the probabilistic Miller–Rabin test. It is not
  a check of every odd number in a range, and it does not extend the record sequences.

We did not find an exhaustive verification of every odd n up to a bound beyond 2·10^13, or record terms beyond
the OEIS tables. That is a statement about what we found, not a proof that none exists.

## Method (`src/lemoine.c`)

- **Exact phase.** For n < 2^28 (B) and n < 2^29 (A), every n is checked one by one against a bit array of the
  odd primes below 2^29.
- **Fast phase.** The range is cut into windows [X, X + W) with W = 2^30. For each window the odd primes in
  [X − 2·16384, X + W) are sieved into a bit array P with primesieve.
  - B: n − 2p is prime for the j-th small prime p exactly when a fixed shift of P has a 1 bit at n.
    ORing the shifts for the first K = 128 primes removes every n with p(n) ≤ 719 at once (AVX2).
    Each remaining n is checked by trying the following small primes in order.
  - A works the same way for n in (2X, 2X + 2W), split by q mod 4 into two bit-shift families.
  - An n with no solution among the small primes in the margin would be reported as ANOMALY. None occurred.
  - Each window keeps only its local chain of records. The chains are merged in window order. Merging is
    valid because a record must exceed the K-th small prime, and this premise is checked when the fast phase
    starts.
- 240 windows form a segment. After each segment the program writes a `DONE` line: the segment [X, E), the
  running prime count, the uncovered counts, the largest values, the anomaly count, and the wall time in seconds.
- Options used for checking: `SELFCHECK=1` recounts the primes of every window with
  `primesieve_count_primes`. `CHECK=1` recomputes 1/1024 of the OR words naively.

The production run used W = 2^30, SEG = 240, K = 128 and 12 threads. It wrote 3,881 `DONE` lines with
anom = 0, then `ALLDONE N=1000000725188608 pi=29844591414555 recA=13037 recB=5881 fail=0`.
The segment times add up to about 71.5 hours.

## Verification

Each check below is a different kind of evidence. Only the production run computed the whole range. Three checks
cover the whole log: the coverage of the `DONE` lines, π(N), and the value at each of the 111 records. The other
checks cover only the part named in the table: the known OEIS terms, 60 segments, and the ranges of the independent
reimplementation. The source identity check is about the two source files, not about any range of n.

| Check | What it shows | Script | Output |
|---|---|---|---|
| Exhaustive run | every n in range, one program | `src/lemoine.c` | `log/l1e15.log` |
| Coverage of the log | the `DONE` lines run from 2^28 to N without gap, widths 240·2^30 (the last one 123·2^30), prime counts add up, anom = 0, `ALLDONE` fail = 0, `EXIT rc=0` | `verify/continuity.py` | `verify/continuity.txt` |
| π(N) | the prime count in the log equals primecount's π(N − 1), and the known π(10^15) plus a primesieve count of [10^15, N − 1] | `verify/pi_check.sh` | `verify/pi_check.txt` |
| OEIS agreement | the first 50 A records and 41 B records equal the OEIS tables | `check_oeis.py` | — |
| Value at each record | for all 111 `REC` lines, p(n) or q(n) recomputed by trial with a deterministic Miller–Rabin test, sharing no code with `src/lemoine.c` | `verify/naive_check.py` | `verify/naive_check.txt` |
| Segment recomputation | 60 randomly chosen segments (seeds 1, 2, 3; 20 each; from 2.6·10^12 to 9.8·10^14) recomputed with `SELFCHECK=1`; every `DONE` line equal to the log except `wall` and the running `pi` (`piseg` is compared). `REC` lines are not compared; the records before each segment are taken from the log | `verify/segments.sh` | `verify/seg_seed*.txt` |
| Source identity | `src/lemoine_en.c` equals `src/lemoine.c` after removing comments | `verify/same_code.sh` | — |
| Independent reimplementation | a separate program written without seeing this code or log (see below) | not included | — |

The independent reimplementation (a segmented sieve in Rust and a Miller–Rabin reference in Python, written by a
second AI system from the definitions alone) was given only the definitions and the 20 new n, and was asked:

- p(n) or q(n) at each of the 20 new n, and the maximum over the 5·10^7 odd m in [n − 10^8, n). All 20 values
  agree with the log; every maximum is smaller than the value at n.
- For four windows [X, X + 2^30) (two contain the largest new A and B records, two are random): the prime count,
  the largest p(n) and q(n) with their first n, and the number of n with p(n) > 719 or with q(n) outside the first
  128 primes of each residue class mod 4. All 28 numbers (7 per window) agree with this program run on the same
  windows.
- All records of p(n) for n < 10^10 (29) and of q(n) for n < 2·10^10 (35). They agree with the log and with OEIS.

Limits:

- The record property (every smaller n has a smaller value) rests on the exhaustive run. Outside the 60
  recomputed segments, the four windows and the ranges below 10^10 and 2·10^10, it was not reproduced by a
  second run. Even inside the 60 segments, the recomputation starts from the records in the log, so it confirms
  the segment but does not re-establish the record property independently.
- The naive check confirms the value at each record n. It does not search between records.
- A negative control was run for `naive_check.py`: a log with two altered records gave two MISMATCH lines.

## Files

- `src/lemoine.c`: the source of the run, byte for byte. Its md5 begins with `babe91bdc003`, the value in the
  `BUILD` line of the log. Its comments are in Japanese and it has CRLF line ends; it is kept unchanged.
- `src/lemoine_en.c`: the same code with the comments translated into English (checked by `verify/same_code.sh`).
- `log/l1e15.log`: the log of the production run. Line types: `BUILD`, `EXACT`, `START`, `REC A n q`, `REC B n p`,
  `DONE`, `ALLDONE`, `EXIT`. It contains no clock times.
- `oeis/`: the OEIS b-file data used by `check_oeis.py` (index, n, value).
- `Dockerfile`: Debian bookworm with gcc and primesieve 11.0. `verify/Dockerfile` adds primecount 7.6.
- `build.sh`, `run_production.sh`: build, and the production run as it was started.

## Reproduce

Requirements: Docker that runs Linux containers on an x86_64 CPU with AVX2 (the program uses AVX2 instructions and
has no fallback, so it does not run on ARM or on CPUs without AVX2), Python 3 on the host, and Bash.
Each script exits with a non-zero status if a check fails.

```bash
docker build -t gilb-env .
docker run --rm -v "$PWD":/w -w /w gilb-env bash build.sh
python check_oeis.py log/l1e15.log 10000000000000
python verify/continuity.py log/l1e15.log
python verify/naive_check.py log/l1e15.log
docker run --rm -v "$PWD":/w -w /w gilb-env bash verify/same_code.sh
docker build -t gilb-env-pc -f verify/Dockerfile verify
docker run --rm -v "$PWD":/w -w /w gilb-env-pc bash verify/pi_check.sh
docker run --rm -e OMP_NUM_THREADS=16 -v "$PWD":/w -w /w gilb-env bash verify/segments.sh 20 1
```

`python` means Python 3; on some systems the command is `python3`, on Windows it may be `py -3`.
On Windows with Git Bash, put `MSYS_NO_PATHCONV=1` before `docker run` and use `"$(pwd -W)"` for `$PWD`.
The binary is built with `-march=native`, so its md5 depends on the CPU; compare the source md5 only.
Set `OMP_NUM_THREADS` to the number of threads you want. Each recomputed segment took about one minute with
16 threads on our machine; `pi_check.sh` takes a few seconds.
The full run (`run_production.sh`) takes about three days on 12 threads.

## How this was made

The program, the production run, the checks and this write-up were produced with the AI assistant Claude
(Anthropic). The independent reimplementation described under Verification was produced with OpenAI's GPT
(Codex). The author directed the work and checked the reports but is not a specialist in this area.
Please treat the result accordingly.

## Author

Hirotaka Shimizu

## License

MIT (see `LICENSE`).
