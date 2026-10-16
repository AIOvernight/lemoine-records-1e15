# GPU extension: how it was made, and credit

This branch adds to Hirotaka Shimizu's repository a GPU continuation of his run, from his end point
N = 1000000725188608 to E = 3607647169282048 (B) and 2E (A). Everything below the line "GPU extension" in the
README, and every file whose name starts with `gpu_` or that lives in `verify/gpu_*`, `src/lemgpu*.cu`,
`src/cpucheck.c`, `build_gpu.sh`, `run_gpu.sh`, `log/gpu_3.6e15.log` and `oeis/candidates_gpu.txt`, was added by this
branch. Shimizu's files are unchanged.

## Provenance

- The GPU program, the production run, the checks and the texts of this branch were produced by an AI agent
  (Claude, Anthropic) running unattended on a desktop machine with one NVIDIA RTX 5090, coordinated by the person
  behind the handle **[@AIOvernight](https://x.com/AIOvernight)**. That person directed the work, reviewed the
  reports and decided what to publish, but is not a specialist in this area and did not check the mathematics
  independently. No human expert has reviewed the result.
- The work builds directly on Shimizu's: the same window grid (X = 2^28 + k·2^30, margin 16384 odd numbers,
  K = 128 small primes, q = 1 excluded), the same statistics in the `DONE` lines, and his log as the calibration
  reference (`verify/gpu_calibration.md`). The GPU program was written from his source and README; it is a new
  implementation, not a port of `lemoine.c` line by line.
- The run was done on 2026-10-15 in one session: 6.31 hours of GPU wall time over the 10119 logged segments.

## Credit and license

Credit for the GPU extension goes to the handle @AIOvernight only. It carries no claim of a mathematical
qualification, affiliation, endorsement or independent human verification. The original work and its credit
remain Hirotaka Shimizu's (see the README sections "How this was made" and "Author").

The added files are released under the same MIT license as the repository (`LICENSE`); the copyright notice of
the original work is kept. The four candidate terms are meant to be submitted to the OEIS with proper credit to
Shimizu's run as the basis; that submission has not been made as of this branch.

## Contact

Issues or pull requests on this repository, or the handle above.
