"""Check that the DONE lines of the log cover [2^28, N) without gaps.
Checks: each X equals the previous E; the width E - X is 240 * 2^30 (the last one may be shorter); the first X is 2^28;
the last E is N of the ALLDONE line; pi of each DONE line equals pi(2^28) + the running sum of piseg; anom is 0 everywhere;
the last width is a positive multiple of 2^30; there is one START line; there is no failure line (ANOMALY, CHECK FAIL,
SELFCHECK FAIL, SIEVE FAIL or FAIL, the forms written by src/lemoine.c); ALLDONE has fail=0; the last non-empty line
is exactly EXIT rc=0.
Usage: python verify/continuity.py log/l1e15.log"""
import sys

W = 240 * 2**30
done = []
alldone_n = None
alldone_pi = None
alldone_fail = None
exit_rc = None
last_line = None
failure_lines = []
FAILURE_PREFIXES = ("ANOMALY", "CHECK FAIL", "SELFCHECK FAIL", "SIEVE FAIL", "FAIL")
other = {}
for line in open(sys.argv[1]):
    f = line.split()
    if not f:
        continue
    last_line = line.strip()
    if line.startswith(FAILURE_PREFIXES):
        failure_lines.append(line.strip())
    if f[0] == "DONE":
        d = dict(x.split("=") for x in f[1:] if "=" in x)
        done.append(d)
    elif f[0] == "ALLDONE":
        d = dict(x.split("=") for x in f[1:] if "=" in x)
        alldone_n = int(d["N"])
        alldone_pi = int(d["pi"])
        alldone_fail = int(d["fail"])
    elif f[0] == "EXIT":
        exit_rc = f[1]
    if f[0] in ("START", "BUILD", "EXIT"):
        other[f[0]] = other.get(f[0], 0) + 1

errors = []
prev_e = 2**28
total = 14630843  # pi(2^28): primes of the exact stage, included in pi of every DONE line (primecount 268435455)
for i, d in enumerate(done):
    x, e = int(d["X"]), int(d["E"])
    if x != prev_e:
        errors.append("#%d: X=%d but previous E=%d" % (i + 1, x, prev_e))
    if e - x != W and i != len(done) - 1:
        errors.append("#%d: width %d != 240*2^30" % (i + 1, e - x))
    if i == len(done) - 1 and not (0 < e - x <= W and (e - x) % 2**30 == 0):
        errors.append("#%d (last): width %d is not a positive multiple of 2^30 up to 240*2^30" % (i + 1, e - x))
    total += int(d["piseg"])
    if int(d["pi"]) != total:
        errors.append("#%d: pi=%s but cumulative piseg=%d" % (i + 1, d["pi"], total))
    if int(d["anom"]) != 0:
        errors.append("#%d: anom=%s" % (i + 1, d["anom"]))
    prev_e = e

print("DONE lines: %d, first X=%s, last E=%d, last width=%d*2^30"
      % (len(done), done[0]["X"], prev_e, (prev_e - int(done[-1]["X"])) // 2**30))
print("ALLDONE N=%s pi=%s fail=%s; EXIT %s; other lines: %s" % (alldone_n, alldone_pi, alldone_fail, exit_rc, other))
if other.get("START", 0) != 1:
    errors.append("START lines: %d (expected 1)" % other.get("START", 0))
for l in failure_lines:
    errors.append("failure line: " + l)
if alldone_fail != 0:
    errors.append("ALLDONE fail=%s (expected 0)" % alldone_fail)
if exit_rc != "rc=0":
    errors.append("EXIT %s (expected rc=0)" % exit_rc)
if last_line != "EXIT rc=0":
    errors.append("last line is %r (expected 'EXIT rc=0')" % last_line)
if prev_e != alldone_n:
    errors.append("last E=%d != ALLDONE N=%s" % (prev_e, alldone_n))
if alldone_pi is not None and total != alldone_pi:
    errors.append("pi(2^28) + sum of piseg=%d != ALLDONE pi=%d" % (total, alldone_pi))
for e in errors[:20]:
    print("  " + e)
print("continuity: " + ("OK" if not errors else "NG (%d issues)" % len(errors)))
sys.exit(0 if not errors else 1)
