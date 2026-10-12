"""Compare the REC lines of the log with the existing OEIS tables.
Usage: python check_oeis.py log/l1e15.log 10000000000000
With the second argument N, only the range of the OEIS tables is compared: B for n < N, A for n < 2N."""
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
log = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else None
got = {"A": [], "B": []}
for line in open(log):
    f = line.split()
    if len(f) == 4 and f[0] == "REC":
        got[f[1]].append((int(f[2]), int(f[3])))
ok = True
for t, fn, lim in (("A", "oeis_A194828.txt", 2 * N if N else None), ("B", "oeis_A195353.txt", N)):
    exp = [(int(a), int(b)) for _, a, b in (l.split() for l in open(os.path.join(root, "oeis", fn)))]
    if lim:
        exp = [e for e in exp if e[0] < lim]
    g = got[t][: len(exp)] if lim else got[t]
    same = g == exp
    ok &= same
    print(f"{t}: {'OK' if same else 'MISMATCH'} expected {len(exp)} records, got {len(got[t])}")
    if not same:
        for i, (e, h) in enumerate(zip(exp, got[t])):
            if e != h:
                print(f"  first diff at #{i+1}: oeis={e} ours={h}")
                break
        if len(got[t]) < len(exp):
            print(f"  ours stops at {got[t][-1] if got[t] else None}")
sys.exit(0 if ok else 1)
