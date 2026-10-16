#!/usr/bin/env python3
"""Summary statistics from the production logs: throughput, per-segment maxima, record values vs (ln n)^2."""
import sys, re, math
segs = []; recs = []
for fn in sys.argv[1:]:
    for line in open(fn):
        if line.startswith("DONE"):
            d = dict(re.findall(r"(\w+)=(\S+)", line)); segs.append(d)
        elif line.startswith("REC"):
            f = line.split(); recs.append((f[1], int(f[2]), int(f[3])))
segs.sort(key=lambda d: int(d["X"]))
tot = sum(int(d["E"]) - int(d["X"]) for d in segs); wall = sum(float(d["wall"]) for d in segs)
print(f"segments {len(segs)}  range [{segs[0]['X']}, {segs[-1]['E']})  numbers {tot:.4g}  wall {wall/3600:.2f} h  mean rate {tot/wall:.3g} n/s")
mb = [int(d["maxB"]) for d in segs]; ma = [int(d["maxA"]) for d in segs]
print(f"per-segment maxB: min {min(mb)} median {sorted(mb)[len(mb)//2]} max {max(mb)};  maxA: min {min(ma)} median {sorted(ma)[len(ma)//2]} max {max(ma)}")
ub = sum(int(d["uncB"]) for d in segs); ua = sum(int(d["uncA"]) for d in segs)
print(f"uncovered fraction (beyond K=128 small primes): B {ub/(tot/2):.4%}  A {ua/(tot):.4%}")
print("records (value / ln(n)^2):")
for k, n, v in sorted(recs, key=lambda r: r[1]):
    print(f"  REC {k} {n} {v}   v/ln(n)^2 = {v/math.log(n)**2:.3f}")
