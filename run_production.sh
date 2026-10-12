#!/bin/bash
# The production run, as it was started (paths adapted to this repository; the run itself used the same source,
# src/lemoine.c, md5 prefix babe91bdc003). It verifies up to N = 2^28 + 931323 * 2^30 = 1000000725188608:
# B for n < N, A for n < 2N. W = 2^30, SEG = 240, K = 128 (the defaults of lemoine).
# The container restarts by itself and lemoine resumes from the DONE lines of the log.
# When lemoine ends, an EXIT line is written and the container sleeps (to avoid a restart loop).
# The full run took about 71.5 hours of computation on 12 threads.
# Usage: bash run_production.sh [CPUS=12]
CPUS=${1:-12}
N=$(( (1<<28) + 931323 * (1<<30) ))
L=log/l1e15.new.log
docker run -d --name lemoine --restart unless-stopped --cpus "$CPUS" -v "$PWD":/w -w /w gilb-env \
  bash -c 'gcc -O3 -march=native -fopenmp -Wall -Wextra -o lemoine src/lemoine.c -lprimesieve || { echo "BUILD FAILED" >> '"$L"'; sleep infinity; }
    echo "BUILD src=$(md5sum src/lemoine.c | cut -c1-12) bin=$(md5sum lemoine | cut -c1-12)" >> '"$L"'
    ./lemoine '"$N"' LOG='"$L"' > /dev/null; rc=$?
    echo "EXIT rc=$rc" >> '"$L"'
    sleep infinity'
