# The CUDA kernels of `src/lemgpu3.cu`

How the GPU continuation of this repository's run is implemented, kernel by kernel, from the source. The pipeline
follows `src/lemoine.c` (Shimizu) step for step on the same window grid and with the same conventions, so that its
`DONE` and `REC` lines are directly comparable with `log/l1e15.log`; the implementation is new.

`src/lemgpu.cu` is the first version (it produced the first 213 `DONE` lines of `log/gpu_3.6e15.log`);
`src/lemgpu3.cu` is the optimised one. Section 8 lists the differences.

## 1. Geometry and conventions (identical to `lemoine.c`)

- **Window grid.** X = 2^28 + k·2^30, W = 2^30. Instance B (least p, OEIS A195353/4) handles the odd n in (X, X + W):
  n = X + 1 + 2t, t in [0, 2^29). Instance A (least q, A194828/9) handles the odd n in (2X, 2X + 2W), split into two
  sub-instances by residue mod 4: n = 2X + 4t + 3 (q ≡ 1 mod 4) and n = 2X + 1 + 4t (q ≡ 3 mod 4). A window therefore
  verifies 2^30 numbers for B and 2^31 for A from a single prime bitmap.
- **Prime bitmap P.** One bit per odd number: bit j is lo + 2j with lo = X − 2·MARG + 1, MARG = 16384, 1 = prime. L = W/2 +
  MARG bits = 2^29 + 16384 (about 64 MB). The margin of MARG odd numbers below X is where n − 2p (B) and (n − q)/2 (A) land
  for the small primes.
- **Shift tables.** For the primes p ≤ MARG (B) and the primes q ≤ 4·MARG + 1 split by q mod 4 (A), the host computes the
  bit offset s so that the partner of n = n(t) is bit t + s: s = MARG − p for B, s = MARG − (q − 1)/4 for q ≡ 1 (mod 4),
  s = MARG − 1 − (q − 3)/4 for q ≡ 3 (mod 4). The three lists are stored in constant memory (`cS[3][4096]`); B contains
  p = 2 as well (n = 2·2 + q, q = n − 4), A contains only odd primes q ≥ 3, so q = 1 is excluded as in `lemoine.c`.
- **K = 128 "fast" primes.** The first K primes of each list define the fast phase. The run's premise, checked at start-up
  (`premise violated` otherwise), is that the running records RECB/RECA are at least the K-th prime of each list
  (719 for B, 1697 and 1567 for the two A lists). Then any n whose partner is prime for one of the first K primes has a
  least prime ≤ 719 (or 1697/1567) and cannot be a record, so only the *uncovered* n need an exact search.

## 2. `k_small`: shared-memory sieve with the 3..31 wheel (primes below 2^20)

```
k_small<<<nblk, 1024, SUB/8>>>(P32, L, lo, primes[37..2^20), nsmall, SUBBITS=19, wheel offsets, usewheel)
```

Each block sieves SUB = 2^19 bits in 64 KB of dynamic shared memory (`__launch_bounds__(1024, 2)`; a window needs 1025
blocks). The block builds a composite mask:

1. primes with more than 256 multiples per block (p ≤ 2048): block-wide loop over the primes, the 1024 threads stride the
   multiples cooperatively (`atomicOr` in shared memory);
2. the other primes below 2^20: one prime per thread, each thread marks all multiples of its prime in the block;
3. the ten primes 3, 5, …, 31 are not sieved: for each, 32 precomputed 32-bit masks (one per residue class, `cWM[10][32]`
   in constant memory, staged in shared memory) are ORed into the output word according to the word's residue, computed
   from the global bit index and the offset of the prime's first odd multiple in the window (`WheelOff.j0`);
4. each output word of P is the complement of the mask, the tail beyond L masked off.

The wheel removes the ten most expensive primes from the atomic loops; the prime list given to the kernel starts at 37.

## 3. `k_large`: global-atomic sieve with mod-30 skipping (primes from 2^20 to √(X + W))

```
k_large<<<⌈n/256⌉, 256>>>(P32, L, lo, primes[2^20..sq), n)
```

One thread per sieving prime (4.16 million primes up to √(5·10^15 + W) in the production run; `sieve_primes=4157406` in
the `START` line), clearing odd multiples with `atomicAnd` on 32-bit words in global memory. Since the wheel already
covers the multiples of 3 and 5, each thread only touches the multiples m·p whose cofactor m is coprime to 30: it walks
m through the residues {1, 7, 11, 13, 17, 19, 23, 29} mod 30 with the gap table {6, 4, 2, 4, 2, 4, 6, 2}, starting at the
first admissible cofactor at or after the prime's first odd multiple in the bitmap (`cyc`/`res` tables). This removes
8/15 of the atomics of the large primes, which are the part of the sieve that grows with √N.

## 4. `k_count`: prime count of the window

A grid-stride popcount of P from bit MARG (the odd number X + 1) to L, warp-shuffle reduced, one `atomicAdd` per warp.
It is the `piseg` of the `DONE` line; summed over the run it is cumulated in `pi` and checked against `primecount`.

## 5. `k_cover`: the 128-shift OR cover in shared-memory tiles

```
k_cover<<<⌈(W/2/32)/256⌉, 256>>>(P, nwords = W/2/32, kind, K, thr, stats, cand, unc_list)      once per instance kind 0, 1, 2
```

Thread w owns 32 consecutive t (one 32-bit output word). An n is *covered* if its partner is prime for at least one of
the first K primes of the instance, i.e. bit t + s_j of P is set for some j < K. The kernel computes, for the 32 t of
the word, the OR of P shifted by the K offsets:

- The block first copies the slice of P it needs into shared memory: a tile of CB + MARG/32 + 8 = 776 words (CB = 256
  output words plus the maximal shift MARG bits), loaded once with `__ldg`.
- The K shifts of the instance are stored in constant memory as (word offset, bit offset) pairs, **sorted by word
  offset** (`cOff`, `cSh`). The loop over j keeps the pair of tile words (a, b) for the current word offset and only
  reloads them when the offset changes; each shift is one `__funnelshift_r(a, b, sh)` (a single instruction extracting
  32 bits across the pair) ORed into the accumulator. Grouping by word offset is what turns 128 unaligned 64-bit reads
  per word into a handful of shared-memory loads.
- `u = ~c` is the mask of uncovered t in the word. The uncovered t (about 0.22 % of all n) are appended to a global list
  with a **warp-aggregated** atomic: a prefix sum of the popcounts across the warp (`__shfl_up_sync`), one `atomicAdd` per
  warp on the list counter (`st->nunc`), then each lane writes its positions at its reserved slots. The list is capped at
  2^23 entries; an overflow is counted as an anomaly (it never happened).

## 6. `k_scan`: the exact phase on the uncovered n

```
k_scan<<<4096, 256>>>(P, unc_list, kind, K, thr, stats, cand)
```

Grid-stride over the list of uncovered t. For each one the thread continues the search where the fast phase stopped: it
tests the primes of the instance in ascending order from index K onwards (up to p ≤ MARG for B, q ≤ 4·MARG + 1 for A),
reading one bit of P per prime, and stops at the first prime partner. The value is converted back from the shift
(v = MARG − s for B, 4(MARG − s) + 1 and 4(MARG − 1 − s) + 3 for the two A lists), the instance maximum is updated with
`atomicMax`, and if v exceeds the record at the start of the window (`thr`) a candidate (t, v, kind) is appended to the
candidate buffer. An n with no prime partner within the margin would be an **ANOMALY**: counted, listed with v = 0 and the
run marked as failed. There were none in 10 119 segments.

`k_fin` then moves the uncovered count into the per-instance statistics and resets the list for the next instance.
The three instances (B, A with q ≡ 1, A with q ≡ 3 mod 4) run this sequence in turn on the same bitmap.

## 7. Record chaining on the host

Candidates are copied back, converted to n with `n_of(kind, X, t)`, sorted by n separately for B and for A (the two A
sub-instances merge into one chain), and walked in order: every value above the running record becomes a `REC B n p` /
`REC A n q` line and raises the record. Because candidates are only those above the record at the window start and are
processed in n order, the chain is exactly the record sequence, and the n printed is the first n attaining the value.
The `DONE` line of a segment (SEG = 240 windows = 240·2^30 numbers, as in `lemoine.c`) reports the cumulative prime count,
the segment prime count, the uncovered counts `uncB`/`uncA`, the maxima `maxB`/`maxA`, the anomaly count and the rate.

## 8. Differences from the first version (`src/lemgpu.cu`)

| | `lemgpu.cu` (first 213 segments) | `lemgpu3.cu` (the rest) |
|---|---|---|
| small-prime sieve | shared-memory blocks of 2^SUBBITS bits (2^19 in production), every prime ≥ 3 sieved by atomics | same blocks, the primes 3..31 by wheel masks instead of atomics, `__launch_bounds__(1024, 2)` |
| large-prime sieve | every odd multiple cleared | mod-30 skipping of the cofactor |
| cover | one thread per 64 t, 128 unaligned 64-bit reads of P from global memory (`__ldg`) per word | one thread per 32 t, shared-memory tile, shifts grouped by word offset, funnel shifts |
| exact phase | inside the cover kernel, each thread scans its own uncovered t serially | separate `k_scan` over a compacted list of uncovered t (warp-aggregated append), load-balanced |
| throughput (`rate=` of the `DONE` lines) | median 5.1 × 10^10 n/s | median 1.37 × 10^11 n/s, up to 1.71 × 10^11; 1.66 × 10^11 in a timing run on an idle GPU |

Both versions were calibrated on the same two segments of Shimizu's grid and give the same `DONE` fields and `REC` lines
(`verify/gpu_calibration.md`).

## 9. Throughput

Per-segment rates over the 10 119 segments of `log/gpu_3.6e15.log`: minimum 3.0 × 10^10, median 1.37 × 10^11, maximum
1.71 × 10^11 numbers per second; 8185 segments above 1.3 × 10^11. The low values are the 213 `lemgpu` segments and the
segments that include the CPU self-checks (a SELFCHECK window builds a primesieve bitmap on the host; a NAIVE window
rescans every n on the CPU), plus the last minutes when the GPU was shared with the final checks. The mean over the whole
run, including everything, was 1.15 × 10^11 numbers per second: 2.6 × 10^15 numbers in 6.31 hours on one RTX 5090
(CUDA 12.8, sm_120). Shimizu's CPU run covered 10^15 numbers in 71.5 hours on 12 threads, 3.9 × 10^9 numbers per second,
so the GPU program is about 37 times faster on the sustained figure and about 43 times on the optimised version's median.

## 10. Self-checks built in

- `SELFCHECK=k`: every k-th window (and the first two), the GPU bitmap is copied back and compared word for word with a
  bitmap built on the host from primesieve, and the prime counts are compared (`SELFCHECK ... words_diff=0 ... ok`).
  10 031 windows in the log, all ok (every 97th window under `lemgpu`, every 251st under `lemgpu3`).
- `NAIVE=k`: every k-th window, the host recomputes the uncovered counts and the maxima of the three instances by a direct
  per-n scan over the primesieve bitmap, independent of the OR trick, and compares them with the GPU's (`NAIVE ... ok`).
  1296 windows, all ok (every 503rd / 2003rd window).
- `PICUM=`: cumulative prime count at `START`, so that a resumed run continues the `pi` column exactly; `RECA=`/`RECB=`:
  the running records (also the premise of the fast phase, section 1).

Independent of the program: `src/cpucheck.c` recomputes whole segments on the CPU (primesieve bitmap, direct per-n scan,
no shared code with the kernels), `verify/gpu_naive_check.py` re-derives the four records from the definitions with
deterministic Miller–Rabin, and Shimizu's own `verify/naive_check.py` accepts the log as is (see the README section
"Verification").

## 11. Build

    nvcc -O3 -arch=sm_120 -Xcompiler -fopenmp src/lemgpu3.cu -o lemgpu3 -lprimesieve

`build_gpu.sh` builds `lemgpu3`, `lemgpu` and `cpucheck` for a given architecture and primesieve prefix; `run_gpu.sh`
holds the production and resume command lines. The program requires `START`/`END` on the window grid and refuses a
window whose margin would fall below √(END + W).
