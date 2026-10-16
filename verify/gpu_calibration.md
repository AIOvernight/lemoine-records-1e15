# Calibration of the GPU program against `log/l1e15.log`

Before the production run, the GPU program was run on two segments of Shimizu's grid and its output compared
with the corresponding lines of `log/l1e15.log`: the `DONE` fields `piseg`, `uncB`, `uncA`, `maxB`, `maxA`, `anom`,
and every `REC` line of the segment (`wall`, `rate` and the running `pi` are not compared; the running `pi` only
agrees when the GPU program is started with `PICUM=` set to the prime count at `START`).

| Segment | GPU program | Output | Reference | Result |
|---|---|---|---|---|
| 1st segment, [2^28, 257966473216) (A: [2^29, 515932946432)) | `lemgpu` (first version) | `gpu_calibration/lemgpu_segment1.txt`, log `lemgpu_segment1.log` | `lemoine_c_segment1.txt` (the 17 `REC` lines of `log/l1e15.log` in that range) | `DONE` fields identical; all 17 `REC` lines (5 B, 12 A) identical |
| 1st segment | `lemgpu3` (optimised; used for 97 % of the production range) | `gpu_calibration/lemgpu3_segment1.txt`, log `lemgpu3_segment1.log` | same | identical |
| last segment, [999868654944256, N) | `lemgpu` | `gpu_calibration/lemgpu_segment_last.log` | last `DONE` line of `log/l1e15.log` | `DONE` fields identical |
| last segment | `lemgpu3` | `gpu_calibration/lemgpu3_segment_last.log` | same | `DONE` fields identical |

The calibration runs of the first segment started with `RECA=3001 RECB=1483`, the records of the exact phase of
`lemoine.c` (n < 2^28 for B, n < 2^29 for A), so the `REC` chain continues the chain of `log/l1e15.log`. The 71
`REC` lines of `log/l1e15.log` with smaller n belong to that exact phase, which the GPU program does not repeat.

Reproduce (after `build_gpu.sh`; each segment takes a few seconds on the GPU):

```bash
./lemgpu3 257966473216 START=268435456 PICUM=14630843 RECA=3001 RECB=1483 SELFCHECK=1 NAIVE=97 LOG=seg1.log
diff <(grep -E "^(DONE|REC)" seg1.log | sed 's/ wall=.*//') \
     <(awk '/^REC B/ && $3>=268435456 && $3<257966473216 || /^REC A/ && $3>=536870912 && $3<515932946432 || /^DONE X=268435456 /' log/l1e15.log | sed 's/ wall=.*//')
./lemgpu3 1000000725188608 START=999868654944256 PICUM=29840767554193 RECA=13037 RECB=5881 SELFCHECK=1 LOG=seglast.log
diff <(grep "^DONE" seglast.log | sed 's/ wall=.*//') <(grep "^DONE X=999868654944256 " log/l1e15.log | sed 's/ wall=.*//')
```

(`PICUM` is the prime count at `START`: π(2^28 − 1) = 14630843, and the `pi` of the preceding `DONE` line for the
last segment.)

In addition, during the production run every 97th (`lemgpu`) or 251st (`lemgpu3`) window the GPU prime bitmap was
compared bit for bit with a primesieve bitmap (`SELFCHECK` lines: 10031 windows, `words_diff=0` in all), and every
503rd / 2003rd window the three uncovered counts and the maxima were recomputed on the CPU by a direct per-n scan
(`NAIVE` lines: 1296 windows, all `ok`). `verify/gpu_continuity.py` fails if any of these lines does not end in `ok`.
