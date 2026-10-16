// cpucheck.c — independent CPU recomputation of per-segment statistics (no shared code with the GPU kernels).
// For each window X in [X0, E): primesieve bitmap of odd numbers in [X-2*MARG+1, X+W); for every odd n, find the least prime
// (B: p with n-2p prime; A: q with (n-q)/2 prime) by direct scanning of the small primes; count n whose least prime is beyond
// the K-th small prime (per instance; A split by q mod 4 as in the GPU/yygotm convention) and the max over those.
// Prints: CPUDONE X= E= piseg= uncB= uncA= maxB= maxA=  and REC-candidates (v > given rec thresholds) as CPUREC lines.
// Build: gcc -O3 -march=native -fopenmp cpucheck.c -o cpucheck -lprimesieve
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <omp.h>
#include <primesieve.h>
#define MARG 16384L
#define W (1ULL << 30)
static int K = 128;
static uint32_t SP[8192], SP1[4096], SP3[4096]; static int NSP, N1, N3;
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: cpucheck X0 E [RECA RECB]\n"); return 1; }
    uint64_t X0 = strtoull(argv[1], 0, 0), E = strtoull(argv[2], 0, 0); unsigned recA = argc > 3 ? atoi(argv[3]) : 13037, recB = argc > 4 ? atoi(argv[4]) : 5881;
    long lim = 4 * MARG + 4; uint8_t *c = calloc(lim + 1, 1);
    for (long i = 2; i * i <= lim; i++) if (!c[i]) for (long j = i * i; j <= lim; j += i) c[j] = 1;
    for (long i = 2; i <= lim; i++) if (!c[i]) { SP[NSP++] = i; if (i % 4 == 1) SP1[N1++] = i; else if (i % 4 == 3) SP3[N3++] = i; }
    const uint32_t thrB = SP[K - 1], thr1 = SP1[K - 1], thr3 = SP3[K - 1];
    const int nwin = (int)((E - X0) / W);
    uint64_t piseg = 0; long uncB = 0, uncA = 0; unsigned maxB = 0, maxA = 0;
    const long L = W / 2 + MARG, LW = L / 64 + 16;
#pragma omp parallel reduction(+:piseg,uncB,uncA) reduction(max:maxB,maxA)
    {
        uint64_t *P = malloc(LW * 8); primesieve_iterator it; primesieve_init(&it);
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nwin; i++) {
            const uint64_t X = X0 + (uint64_t)i * W, lo = X - 2 * MARG + 1, hi = X + W;
            memset(P, 0, LW * 8);
            primesieve_jump_to(&it, lo, hi); uint64_t p, cnt = 0;
            while ((p = primesieve_next_prime(&it)) < hi) { uint64_t j = (p - lo) / 2; P[j >> 6] |= 1ULL << (j & 63); cnt += p > X; }
            piseg += cnt;
#define ISP(num) ((P[(((num) - lo) / 2) >> 6] >> ((((num) - lo) / 2) & 63)) & 1)
            // B: n odd in (X, X+W)
            for (uint64_t n = X + 1; n < X + W; n += 2) {
                int j = 0; for (; j < NSP; j++) { uint64_t q = n - 2 * (uint64_t)SP[j]; if (ISP(q)) break; }
                if (j >= NSP) { printf("CPUANOMALY B n=%" PRIu64 "\n", n); continue; }
                if (SP[j] > thrB) { uncB++; if (SP[j] > maxB) maxB = SP[j]; if (SP[j] > recB) printf("CPUREC B %" PRIu64 " %u\n", n, SP[j]); }
            }
            // A per residue class of q (A0: n = 3 mod 4 uses q = 1 mod 4; A1: n = 1 mod 4 uses q = 3 mod 4)
            for (uint64_t n = 2 * X + 1; n < 2 * X + 2 * W; n += 2) {
                const uint32_t *cls = (n % 4 == 3) ? SP1 : SP3; const int ncls = (n % 4 == 3) ? N1 : N3; const uint32_t thr = (n % 4 == 3) ? thr1 : thr3;
                int j = 0; for (; j < ncls; j++) { uint64_t pp = (n - cls[j]) / 2; if (ISP(pp)) break; }
                if (j >= ncls) { printf("CPUANOMALY A n=%" PRIu64 " (class)\n", n); continue; }
                if (cls[j] > thr) { uncA++; if (cls[j] > maxA) maxA = cls[j]; if (cls[j] > recA) printf("CPUREC A %" PRIu64 " %u\n", n, cls[j]); }
            }
        }
        primesieve_free_iterator(&it); free(P);
    }
    printf("CPUDONE X=%" PRIu64 " E=%" PRIu64 " piseg=%" PRIu64 " uncB=%ld uncA=%ld maxB=%u maxA=%u\n", X0, E, piseg, uncB, uncA, maxB, maxA);
    return 0;
}
