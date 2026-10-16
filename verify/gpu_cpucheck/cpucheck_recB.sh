#!/bin/bash
for n in 1880662361770607 2423130987173003; do
  SEG=$(awk -v n=$n '/^DONE/{split($2,a,"=");split($3,b,"="); if (a[2] < n && n < b[2]) print a[2], b[2]}' prod3.log)
  RB=$(awk -v n=$n '/^REC B/ && $3 < n {v=$4} END{print (v==""?5881:v)}' prod3.log)
  echo "segment for $n: $SEG recB_before=$RB"
  nice -n 10 ./cpucheck $SEG 13037 $RB
  echo "GPU: $(grep "^DONE X=${SEG%% *} " prod3.log | cut -d' ' -f2-9)"
done
echo CPUCHECKS_DONE
