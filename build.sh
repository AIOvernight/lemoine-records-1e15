#!/bin/bash
# Build inside the container (the repository is mounted at /w). Prints the first 12 hex digits of the source md5.
# Usage: docker run --rm -v "$PWD":/w -w /w gilb-env bash build.sh
set -e
gcc -O3 -march=native -fopenmp -Wall -Wextra -o lemoine src/lemoine.c -lprimesieve
echo "built lemoine src=$(md5sum src/lemoine.c | cut -c1-12)"
