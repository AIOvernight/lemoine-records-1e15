#!/bin/bash
# Recompute K randomly chosen segments (DONE lines of the log) with SELFCHECK=1 and compare with the log.
# A segment is [X, E) for B and [2X, 2E) for A. The records found before X are passed as RECA / RECB.
# The pi= field is the running total in the log and the segment total in a recomputation, so it is not compared
# (piseg is). The wall= field (seconds) is not compared. REC lines are not compared: the records before X are
# taken from the log, so this check does not re-establish the record property on its own.
# Prints one "seg X=... OK" line per segment. Exits with 1 (and prints "segments: NG ...") if the program fails,
# a DONE line differs, an ANOMALY / SELFCHECK FAIL line appears, or fewer than K segments were checked.
# Usage (after build.sh): docker run --rm -v "$PWD":/w -w /w gilb-env bash verify/segments.sh 20 1
set -uo pipefail
L=log/l1e15.log; K=${1:-20}; SEED=${2:-1}      # SEED selects a different random sample
ok=0; bad=0
while read -r line; do
  X=$(echo "$line" | sed "s/.*X=\([0-9]*\).*/\1/"); E=$(echo "$line" | sed "s/.* E=\([0-9]*\).*/\1/")
  RA=$(grep "^REC A" $L | awk -v x="$X" '$3<2*x{v=$4} END{print v}'); RB=$(grep "^REC B" $L | awk -v x="$X" '$3<x{v=$4} END{print v}')
  run=$(./lemoine "$E" START="$X" RECA="$RA" RECB="$RB" SELFCHECK=1); rc=$?
  [ "$rc" = 0 ] || { echo "seg X=$X: program exited with $rc"; bad=$((bad+1)); }
  nan=$(echo "$run" | grep -c -E "^ANOMALY|^SELFCHECK FAIL"); [ "$nan" = 0 ] || { echo "seg X=$X: $nan ANOMALY/SELFCHECK lines on recompute"; bad=$((bad+1)); }
  got=$(echo "$run" | grep "^DONE" | sed "s/ wall=.*//; s/ pi=[0-9]*//"); exp=$(echo "$line" | sed "s/ wall=.*//; s/ pi=[0-9]*//")
  if [ "$got" = "$exp" ]; then echo "seg X=$X OK"; ok=$((ok+1)); else echo "seg X=$X MISMATCH"; echo " log: $exp"; echo " now: $got"; bad=$((bad+1)); fi
done < <(grep "^DONE" $L | awk -v s="$SEED" 'BEGIN{srand(s)} {print rand() "\t" $0}' | sort -n | awk -v k="$K" 'NR<=k' | cut -f2-)
if [ "$bad" != 0 ] || [ "$ok" != "$K" ]; then echo "segments: NG ($ok OK of $K, $bad problems)"; exit 1; fi
