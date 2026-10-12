# Lemoine's conjecture: the two record sequences extended to 10^15

Prepared 2026/10/13 by Hirotaka Shimizu with the AI assistant Claude (see How this was made).
Status: computational result. No human expert has checked it yet.

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
