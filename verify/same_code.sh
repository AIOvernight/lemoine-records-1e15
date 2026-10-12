#!/bin/bash
# src/lemoine.c is the source used in the run, byte for byte (md5 prefix babe91bdc003, as in the BUILD line of the log).
# Its comments are in Japanese. src/lemoine_en.c differs only in the comments, which are translated into English.
# This script checks the md5 prefix against the BUILD line, strips the comments of both files with the C
# preprocessor (gcc -fpreprocessed) and compares the two results byte for byte. Exits with a non-zero status on any
# difference or if gcc fails. This shows that these two files have the same code; neither file uses __LINE__, __FILE__ or
# #line, so the translated comments cannot change what the program does.
# Usage: docker run --rm -v "$PWD":/w -w /w gilb-env bash verify/same_code.sh
set -euo pipefail
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
m=$(md5sum src/lemoine.c | cut -c1-12); g=$(grep '^BUILD' log/l1e15.log | sed 's/.*src=\([0-9a-f]*\).*/\1/')
if [ "$m" = "$g" ]; then r=OK; else r=MISMATCH; fi
echo "src/lemoine.c md5 prefix: $m (log: $g) $r"
[ "$r" = OK ] || exit 1
tr -d '\r' < src/lemoine.c    | gcc -fpreprocessed -dD -E -P -x c - -o "$T/a.c"
tr -d '\r' < src/lemoine_en.c | gcc -fpreprocessed -dD -E -P -x c - -o "$T/b.c"
[ -s "$T/a.c" ] && [ -s "$T/b.c" ] || { echo "gcc produced no output"; exit 1; }
if cmp -s "$T/a.c" "$T/b.c"; then r=SAME; else r=DIFFERENT; fi
echo "code without comments: lemoine.c=$(md5sum < "$T/a.c" | cut -c1-12) lemoine_en.c=$(md5sum < "$T/b.c" | cut -c1-12) $r"
[ "$r" = SAME ]
