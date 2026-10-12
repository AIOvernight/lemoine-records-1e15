"""Check every record (n, v) in the REC lines by a naive method that shares no code with src/lemoine.c.
  B (least p): try primes p in increasing order; the first p with n - 2p prime must be v.
  A (least q): try odd primes q in increasing order; the first q with (n - q)/2 prime must be v.
Primality: Miller-Rabin with the first 13 primes as bases, deterministic for n < 3.317e24 (Sorenson and Webster 2015).
A number found composite has a witness, so it is certainly composite.
This checks the value at each n only. It does not check that the value is a record (that all smaller n have smaller
values); that part rests on the main computation.
Usage: python verify/naive_check.py log/l1e15.log"""
import sys

BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
LIMIT = 3317044064679887385961981


def is_prime(n):
    if n < 2:
        return False
    for b in BASES:
        if n % b == 0:
            return n == b
    if n >= LIMIT:
        raise ValueError("outside the deterministic range")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for b in BASES:
        x = pow(b, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def least_p(n):
    p = 2
    while True:
        if is_prime(p) and is_prime(n - 2 * p):
            return p
        p += 1


def least_q(n):
    q = 3
    while True:
        if is_prime(q) and is_prime((n - q) // 2):
            return q
        q += 2


recs = {"A": [], "B": []}
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) == 4 and f[0] == "REC":
        recs[f[1]].append((int(f[2]), int(f[3])))

bad = 0
for t, fn in (("A", least_q), ("B", least_p)):
    ok = 0
    for n, v in recs[t]:
        got = fn(n)
        if n % 2 == 1 and got == v:
            ok += 1
        else:
            bad += 1
            print("%s n=%d log=%d naive=%d MISMATCH" % (t, n, v, got))
    vals = [v for _, v in recs[t]]
    ns = [n for n, _ in recs[t]]
    inc = all(a < b for a, b in zip(vals, vals[1:])) and all(a < b for a, b in zip(ns, ns[1:]))
    print("%s: %d/%d match; n and value strictly increasing: %s" % (t, ok, len(recs[t]), inc))
    if not inc:
        bad += 1
print("naive check: " + ("OK" if bad == 0 else "NG"))
sys.exit(0 if bad == 0 else 1)
