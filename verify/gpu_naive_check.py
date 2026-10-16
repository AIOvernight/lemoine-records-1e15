#!/usr/bin/env python3
"""Independent check of REC lines: deterministic Miller-Rabin (bases 2..37 — correct for n < 3.3e24), no shared code with lemgpu.
For 'REC B n p': checks p prime, n-2p prime, and no smaller prime p' has n-2p' prime.
For 'REC A n q': checks q prime, (n-q)/2 prime, and no smaller prime q' has (n-q')/2 prime.
Also recomputes the value at all odd m in [n-NB, n) and reports the max (must be < value at n for a record)."""
import sys
BASES = [2,3,5,7,11,13,17,19,23,29,31,37]
def is_prime(n):
    if n < 2: return False
    for p in BASES:
        if n % p == 0: return n == p
    d, s = n-1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in BASES:
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True
small = [p for p in range(2, 70000) if is_prime(p)]
def pB(n):   # least prime p with n-2p prime
    for p in small:
        if 2*p >= n: return None
        if is_prime(n-2*p): return p
def qA(n):   # least prime q with (n-q)/2 prime
    for q in small:
        if q >= n: return None
        if (n-q) % 2 == 0 and is_prime((n-q)//2): return q
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
bad = 0
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4 or f[0] != 'REC': continue
    kind, n, v = f[1], int(f[2]), int(f[3])
    fn = pB if kind == 'B' else qA
    got = fn(n)
    mx = 0
    for m in range(max(7, n-NB), n, 2):
        r = fn(m)
        if r is None: print("COUNTEREXAMPLE?", kind, m); bad += 1; continue
        mx = max(mx, r)
    ok = (got == v) and (mx < v)
    bad += not ok
    print(f"{kind} n={n} v={v} recomputed={got} max_prev_{NB}={mx} {'ok' if ok else 'MISMATCH'}", flush=True)
print("FAIL" if bad else "ALL OK", bad)
