#!/usr/bin/env python3
"""Check that the DONE lines of the given logs (in order) cover [START, END) without gap or overlap, that pi is cumulative,
that no ANOMALY/FAIL/OVERFLOW lines exist, and list REC lines in n order with the record property across the logs."""
import sys, re
START = 1000000725188608; PI0 = 29844591414555
logs = sys.argv[1:]
segs = []; recs = []; bad = 0
for fn in logs:
    for line in open(fn):
        if line.startswith("DONE"):
            d = dict(re.findall(r"(\w+)=(\S+)", line)); segs.append((int(d["X"]), int(d["E"]), int(d["pi"]), int(d["piseg"]), int(d["anom"]), fn))
        elif line.startswith("REC"):
            f = line.split(); recs.append((f[1], int(f[2]), int(f[3]), fn))
        elif re.match(r"(ANOMALY|.*FAIL|.*OVERFLOW)", line): print("BAD LINE:", fn, line.strip()); bad += 1
        elif line.startswith("SELFCHECK") or line.startswith("NAIVE"):
            if not line.rstrip().endswith("ok"): print("CHECK NOT OK:", line.strip()); bad += 1
segs.sort()
x = START; pi = PI0; nsc = 0
for (X, E, p, ps, an, fn) in segs:
    if X != x: print(f"GAP/OVERLAP at {x} -> {X} ({fn})"); bad += 1
    if p != pi + ps: print(f"PI mismatch at X={X}: {p} != {pi}+{ps}"); bad += 1
    if an: print(f"ANOM at X={X}"); bad += 1
    x = E; pi = p
nself = sum(1 for fn in logs for l in open(fn) if l.startswith("SELFCHECK")); nnaive = sum(1 for fn in logs for l in open(fn) if l.startswith("NAIVE"))
print(f"segments={len(segs)} covered [{START}, {x})  pi(end-1)={pi}  selfchecks={nself} naive={nnaive}")
ra = 13037; rb = 5881
for kind in "BA":
    rs = sorted([r for r in recs if r[0] == kind], key=lambda r: r[1]); cur = rb if kind == "B" else ra
    for (_, n, v, fn) in rs:
        if v <= cur: print(f"NOT A RECORD: {kind} {n} {v} (current {cur})"); bad += 1
        cur = v; print(f"REC {kind} {n} {v}")
print("CONTINUITY FAIL" if bad else "CONTINUITY OK", bad)
