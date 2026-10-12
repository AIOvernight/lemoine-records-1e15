#!/bin/bash
# Check pi(N) from the ALLDONE line with primecount (combinatorial algorithm, independent of the sieve).
# Also checks the known value pi(10^15) = 29844570422669 (OEIS A006880) and the sum
# pi(10^15) + (primes in [10^15, N-1] counted by primesieve).
# Exits with a non-zero status if a command fails or any line says MISMATCH.
# Usage: docker build -t gilb-env-pc -f verify/Dockerfile verify
#        docker run --rm -v "$PWD":/w -w /w gilb-env-pc bash verify/pi_check.sh
set -euo pipefail
L=log/l1e15.log; bad=0
primecount --version | sed -n 1p
N=$(grep "^ALLDONE" $L | tail -1 | sed "s/.*N=\([0-9]*\).*/\1/")
PI=$(grep "^ALLDONE" $L | tail -1 | sed "s/.*pi=\([0-9]*\).*/\1/")
[ -n "$N" ] && [ -n "$PI" ] || { echo "no ALLDONE line in $L"; exit 1; }
K=$(primecount 1e15)
if [ "$K" = 29844570422669 ]; then r=OK; else r=MISMATCH; bad=1; fi
echo "pi(10^15): primecount=$K known=29844570422669 $r"
P=$(primecount $((N-1)))
if [ "$PI" = "$P" ]; then r=OK; else r=MISMATCH; bad=1; fi
echo "pi(N-1), N=$N: log=$PI primecount=$P $r"
D=$(primesieve 1000000000000000 $((N-1)) -q); S=$((29844570422669 + D))
if [ "$S" = "$PI" ]; then r=OK; else r=MISMATCH; bad=1; fi
echo "primesieve [10^15, N-1] = $D; pi(10^15) + this = $S $r"
exit $bad
