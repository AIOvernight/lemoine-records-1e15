// lemoine.c - exhaustive check of Lemoine's conjecture (every odd n > 5 is n = 2p + q with p, q prime) and extension of two record sequences.
//   A: records of the least q (the term not doubled): OEIS A194828 (n) / A194829 (q). Up to 2*10^13 in 2011.
//   B: records of the least p (the doubled term):     OEIS A195353 (n) / A195354 (p). Up to 10^13 in 2011.
// Method: build the bit array P of odd primes in the interval [X, X+W) with primesieve (with a margin of MARG odd numbers before X).
//   B: for odd n in (X, X+W), the least p such that q = n - 2p is prime.
//   A: for odd n in (2X, 2X+2W), the least q such that p = (n - q)/2 is prime (split into two parts by q = 1, 3 (mod 4)).
//   Both: OR copies of P shifted by the small primes; this removes at once every n solved by one of the first K primes; only the rest is checked one by one.
//   The OR is split into 8 groups by the low 3 bits of the shift (bit offset within a byte); each group ORs byte-aligned loads, then shifts once.
//   An n with no solution (a candidate counterexample) is counted as ANOMALY and the program ends with rc 2.
// Each interval collects only its local chain of records; the chains are joined in interval order into the global records (an n solved by a small prime
//   cannot be a record; the premise 'global record >= K-th prime' is checked at the start of the fast phase).
// Build: gcc -O3 -march=native -fopenmp -Wall -Wextra -o lemoine lemoine.c -lprimesieve
// Usage: lemoine N [W=2^30] [SEG=240] [K=128] [LOG=file] [START=X RECA=q RECB=p] [EXACT=1] [DEBUG=1] [SELFCHECK=1] [CHECK=1]
//   N is the upper limit (B: n < N, A: n < 2N). N and START are XE + (multiple of W). With LOG, the run resumes from the DONE lines.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
#include <primesieve.h>

#define MARG 16384L                       // margin before X (number of odd numbers). B: p <= MARG, A: q <= 4*MARG + 1
#define XE ((uint64_t)1 << 28)            // upper end of the exact phase (every n checked one by one). The fast phase starts at X = XE
static uint64_t W = 1ULL << 30;           // interval width (number of integers, a multiple of 512). Interval boundaries are XE + kW
static int K = 128, SEG = 240;           // SEG: number of windows per segment (a common multiple of 10 and 12)
static int EXACT = 0, DEBUG = 0, SELFCHECK = 0, CHECK = 0;   // CHECK=1: in 1/1024 of the words, recompute the OR naively and compare
static FILE *logf;

static void *must(void *q, const char *what) { if (!q) { fprintf(stderr, "out of memory (%s)\n", what); _exit(3); } return q; }
typedef struct { uint64_t n; uint32_t v; } rec_t;
typedef struct { rec_t *a; int n, cap; } chain_t;
static void push(chain_t *c, uint64_t n, uint32_t v) {
    if (c->n == c->cap) { c->cap = c->cap ? 2 * c->cap : 64; c->a = must(realloc(c->a, c->cap * sizeof(rec_t)), "chain"); }
    c->a[c->n++] = (rec_t){n, v};
}
static void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char *fmt, ...) {
    va_list ap; char buf[512];
    va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    fputs(buf, stdout); fflush(stdout);
    if (logf) { fputs(buf, logf); if (fflush(logf) || ferror(logf)) { fprintf(stderr, "LOG write failed (%s)\n", strerror(errno)); _exit(3); } }
}

// small primes: B uses 2, 3, 5, ...; A is split into q = 1 (mod 4) and q = 3 (mod 4)
static uint32_t *SP, *SP1, *SP3; static int NSP, N1, N3;
static void small_primes(void) {
    long lim = 4 * MARG + 4;
    uint8_t *c = must(calloc(lim + 1, 1), "small primes");
    for (long i = 2; i * i <= lim; i++) if (!c[i]) for (long j = i * i; j <= lim; j += i) c[j] = 1;
    SP = malloc(lim * 4); SP1 = malloc(lim * 4); SP3 = malloc(lim * 4);
    for (long i = 2; i <= lim; i++) if (!c[i]) { SP[NSP++] = i; if (i % 4 == 1) SP1[N1++] = i; else if (i % 4 == 3) SP3[N3++] = i; }
    free(c);
}

// instance: shifts S[j] (index into P = output index t + S[j]) and the map from t to n
typedef struct { char name; const uint32_t *pr; long *S; int nS; int kind; long *off[8]; int cnt[8]; } inst_t;   // kind 0: B, 1: A0 (q=1 mod 4), 2: A1 (q=3 mod 4)
static inst_t INST[3];
static inline uint64_t n_of(const inst_t *I, uint64_t X, uint64_t t) {
    return I->kind == 0 ? X + 1 + 2 * t : I->kind == 1 ? 2 * X + 4 * t + 3 : 2 * X + 1 + 4 * t;
}
static void make_inst(void) {
    // B: n = X+1+2t, q = n-2p <-> P[t + MARG - p]
    // A0: q = 4s+1, n = 2X+4t+3, p = (n-q)/2 <-> P[t + MARG - s]
    // A1: q = 4s+3, n = 2X+1+4t, p = (n-q)/2 <-> P[t + MARG - 1 - s]
    const uint32_t *pr[3] = {SP, SP1, SP3}; int np[3] = {NSP, N1, N3};
    for (int k = 0; k < 3; k++) {
        inst_t *I = &INST[k]; I->name = "B01"[k]; I->kind = k; I->pr = pr[k]; I->S = malloc(np[k] * sizeof(long)); I->nS = 0;
        for (int j = 0; j < np[k]; j++) {
            long s = k == 0 ? MARG - (long)pr[k][j] : k == 1 ? MARG - (long)(pr[k][j] - 1) / 4 : MARG - 1 - (long)(pr[k][j] - 3) / 4;
            if (s < 0) break;
            I->S[I->nS++] = s;
        }
        if (I->nS < K) { fprintf(stderr, "K=%d too large for instance %c (nS=%d)\n", K, I->name, I->nS); exit(1); }
        for (int b = 0; b < 8; b++) { I->off[b] = malloc(K * sizeof(long)); I->cnt[b] = 0; }
        for (int j = 0; j < K; j++) { const int b = I->S[j] & 7; I->off[b][I->cnt[b]++] = I->S[j] >> 3; }   // the first K shifts, grouped by bit offset
    }
}

typedef struct { uint64_t cnt; long unc[3]; long anom; uint32_t mx[3]; } wstat_t;

// Process one instance over one interval. P: L = W/2 + MARG bits + padding. Output T = W/2 bits.
// OR of group b over 'the 32 bytes of chunk w (words w..w+3)': Y_b(w) = OR_i loadu(Pb + w*8 + off_i). Two dependency chains.
static inline __m256i group_or(const uint8_t *base, const long *off, int cn) {
    __m256i y0 = _mm256_setzero_si256(), y1 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 1 < cn; i += 2) {
        y0 = _mm256_or_si256(y0, _mm256_loadu_si256((const __m256i *)(base + off[i])));
        y1 = _mm256_or_si256(y1, _mm256_loadu_si256((const __m256i *)(base + off[i + 1])));
    }
    if (i < cn) y0 = _mm256_or_si256(y0, _mm256_loadu_si256((const __m256i *)(base + off[i])));
    return _mm256_or_si256(y0, y1);
}
// Compute the cover c of chunk w by the naive method (the form used up to round 1), for the CHECK=1 comparison
static __m256i cover_naive(const uint8_t *Pb, const inst_t *I, long w) {
    __m256i c = _mm256_setzero_si256();
    for (int j = 0; j < K; j++) {
        const long idx = w * 64 + I->S[j]; const uint8_t *p = Pb + (idx >> 3); const int b = idx & 7;
        __m256i v = _mm256_loadu_si256((const __m256i *)p), v2 = _mm256_loadu_si256((const __m256i *)(p + 8));
        c = _mm256_or_si256(c, _mm256_or_si256(_mm256_srl_epi64(v, _mm_cvtsi32_si128(b)), _mm256_sll_epi64(v2, _mm_cvtsi32_si128(64 - b))));
    }
    return c;
}

static void run_inst(const uint64_t *P, const inst_t *I, uint64_t X, chain_t *ch, wstat_t *st) {
    const long T = W / 2, nwords = T / 64;
    const uint8_t *Pb = (const uint8_t *)P;
    long unc = 0, anom = 0; uint32_t mx = 0;
    // idx = w*64 + S[j] => byte position w*8 + (S[j]>>3), bit offset S[j]&7. Shifting commutes with OR, so OR per group, then shift once.
    // The read 8 bytes ahead (bits entering from the next word by the shift) is built in registers from Y_b(w+4) of the next chunk:
    //   bytes 8..39 of [Y_b(w) | Y_b(w+4)] = alignr(permute2x128(Y(w), Y(w+4), 0x21), Y(w), 8).
    //   The read-ahead of the last chunk falls in the padding of P (cleared by memset, +16 words).
    __m256i Y[8] = {0};
    if (!EXACT) for (int b = 0; b < 8; b++) Y[b] = I->cnt[b] ? group_or(Pb, I->off[b], I->cnt[b]) : _mm256_setzero_si256();
    for (long w = 0; w < nwords; w += 4) {
        __m256i c = _mm256_setzero_si256();
        if (!EXACT) {
            const uint8_t *nb = Pb + (w + 4) * 8;
            for (int b = 0; b < 8; b++) {
                const int cn = I->cnt[b];
                if (!cn) continue;
                const __m256i y = Y[b];
                if (b == 0) { c = _mm256_or_si256(c, y); Y[0] = group_or(nb, I->off[0], cn); continue; }   // offset 0 needs no read-ahead
                const __m256i yn = group_or(nb, I->off[b], cn);
                const __m256i a1 = _mm256_alignr_epi8(_mm256_permute2x128_si256(y, yn, 0x21), y, 8);
                c = _mm256_or_si256(c, _mm256_or_si256(_mm256_srl_epi64(y, _mm_cvtsi32_si128(b)), _mm256_sll_epi64(a1, _mm_cvtsi32_si128(64 - b))));
                Y[b] = yn;
            }
            if (CHECK && ((w >> 2) & 1023) == 0) {
                const __m256i r = cover_naive(Pb, I, w), d = _mm256_xor_si256(r, c);
                if (!_mm256_testz_si256(d, d)) { out("CHECK FAIL %c X=%" PRIu64 " w=%ld\n", I->name, X, w); _exit(4); }
            }
        }
        uint64_t cw[4]; _mm256_storeu_si256((__m256i *)cw, c);
        for (int k = 0; k < 4; k++) {
            uint64_t u = ~cw[k];
            while (u) {
                const int bi = __builtin_ctzll(u); u &= u - 1;
                const long t = (w + k) * 64 + bi;
                unc++;
                uint32_t v = 0;
                for (int j = EXACT ? 0 : K; j < I->nS; j++) { const long idx = t + I->S[j]; if (P[idx >> 6] >> (idx & 63) & 1) { v = I->pr[j]; break; } }
                if (!v) { anom++; out("ANOMALY %c n=%" PRIu64 " (no prime within margin)\n", I->name, n_of(I, X, t)); continue; }
                if (v > mx) mx = v;
                if (ch->n == 0 || v > ch->a[ch->n - 1].v) push(ch, n_of(I, X, t), v);
            }
        }
    }
    st->unc[I->kind] = unc; st->anom += anom; st->mx[I->kind] = mx;
}

static void do_window(uint64_t X, primesieve_iterator *it, uint64_t *P, chain_t *chB, chain_t *chA, wstat_t *st) {
    const uint64_t lo = X - 2 * MARG + 1, hi = X + W;         // lo is odd and >= 3
    const long L = W / 2 + MARG, LW = L / 64 + 16;
    memset(P, 0, LW * 8); memset(st, 0, sizeof *st);
    primesieve_jump_to(it, lo, hi);
    uint64_t p, cnt = 0;
    uint64_t acc = 0, cw = 0;                                  // avoid a read-write chain on the same word (primes come in increasing order)
    while ((p = primesieve_next_prime(it)) < hi) {
        const uint64_t j = (p - lo) / 2, wj = j >> 6;
        if (wj != cw) { P[cw] |= acc; acc = 0; cw = wj; }
        acc |= 1ULL << (j & 63); cnt += p > X;
    }
    P[cw] |= acc;
    if (p == PRIMESIEVE_ERROR) { out("SIEVE FAIL X=%" PRIu64 " (primesieve error)\n", X); _exit(5); }   // do not report a sieve that stopped midway as a completed window
    st->cnt = cnt;
    if (SELFCHECK) { uint64_t c2 = primesieve_count_primes(X + 1, hi - 1); if (c2 != cnt) { out("SELFCHECK FAIL X=%" PRIu64 " cnt=%" PRIu64 " primesieve=%" PRIu64 "\n", X, cnt, c2); st->anom++; } }
    run_inst(P, &INST[0], X, chB, st);
    run_inst(P, &INST[1], X, chA, st);
    chain_t chA1 = {0};
    run_inst(P, &INST[2], X, &chA1, st);
    // merge the A0 and A1 chains in order of n into one (after merging, keep only v larger than the previous)
    chain_t m = {0}; int i = 0, j = 0;
    while (i < chA->n || j < chA1.n) {
        rec_t r = (j >= chA1.n || (i < chA->n && chA->a[i].n < chA1.a[j].n)) ? chA->a[i++] : chA1.a[j++];
        if (m.n == 0 || r.v > m.a[m.n - 1].v) push(&m, r.n, r.v);
    }
    free(chA->a); free(chA1.a); *chA = m;
}

// Exact phase: check every n < XE (B) and n < 2XE (A) one by one. Primality from the bit array of odd numbers in [0, 2XE).
static void exact_phase(uint64_t *recA, uint64_t *recB, uint64_t *pi_cum) {
    const uint64_t M = 2 * XE; uint64_t *Q = must(calloc(M / 128 + 2, 8), "exact bitmap");
    primesieve_iterator it; primesieve_init(&it); primesieve_jump_to(&it, 3, M);
    uint64_t p, cnt = 1;   // count 2
    while ((p = primesieve_next_prime(&it)) < M) { Q[p / 128] |= 1ULL << ((p / 2) & 63); cnt += p < XE; }
    if (p == PRIMESIEVE_ERROR) { out("SIEVE FAIL exact phase (primesieve error)\n"); _exit(5); }
    primesieve_free_iterator(&it);
    *pi_cum = cnt;
#define ODDPRIME(x) ((x) >= 3 && (Q[(x) / 128] >> (((x) / 2) & 63) & 1))
    for (uint64_t n = 7; n < XE; n += 2) {              // B: least p
        uint32_t v = 0;
        for (int j = 0; j < NSP; j++) { const uint64_t q = n - 2 * (uint64_t)SP[j]; if (n < 2 * (uint64_t)SP[j] + 3) break; if (ODDPRIME(q)) { v = SP[j]; break; } }
        if (!v) { out("ANOMALY B n=%" PRIu64 " (exact phase)\n", n); exit(2); }
        if (v > *recB) { *recB = v; out("REC B %" PRIu64 " %u\n", n, v); }
    }
    for (uint64_t n = 7; n < M; n += 2) {               // A: least q (p = 2 is also allowed)
        uint32_t v = 0;
        for (int j = 1; j < NSP; j++) { if (n < SP[j] + 4) break; const uint64_t pp = (n - SP[j]) / 2; if (pp == 2 || (pp & 1 && ODDPRIME(pp))) { v = SP[j]; break; } }
        if (!v) { out("ANOMALY A n=%" PRIu64 " (exact phase)\n", n); exit(2); }
        if (v > *recA) { *recA = v; out("REC A %" PRIu64 " %u\n", n, v); }
    }
    free(Q);
}

static uint64_t parse_u64(const char *s) { return strtoull(s, NULL, 0); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: lemoine N [W=] [SEG=] [K=] [LOG=] [START= RECA= RECB=] [EXACT=1] [DEBUG=1] [SELFCHECK=1] [CHECK=1]\n"); return 1; }
    int fail = 0;
    uint64_t N = parse_u64(argv[1]), X0 = 0, recA = 0, recB = 0, pi_cum = 0; const char *logname = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strncmp(argv[i], "W=", 2)) W = parse_u64(argv[i] + 2);
        else if (!strncmp(argv[i], "SEG=", 4)) SEG = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "K=", 2)) K = atoi(argv[i] + 2);
        else if (!strncmp(argv[i], "LOG=", 4)) logname = argv[i] + 4;
        else if (!strncmp(argv[i], "START=", 6)) X0 = parse_u64(argv[i] + 6);
        else if (!strncmp(argv[i], "RECA=", 5)) recA = parse_u64(argv[i] + 5);
        else if (!strncmp(argv[i], "RECB=", 5)) recB = parse_u64(argv[i] + 5);
        else if (!strcmp(argv[i], "EXACT=1")) EXACT = 1;
        else if (!strcmp(argv[i], "DEBUG=1")) DEBUG = 1;
        else if (!strcmp(argv[i], "SELFCHECK=1")) SELFCHECK = 1;
        else if (!strcmp(argv[i], "CHECK=1")) CHECK = 1;
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    if (W % 512 || N < XE || (N - XE) % W || (X0 && (X0 < XE || (X0 - XE) % W))) {
        fprintf(stderr, "W must be a multiple of 512; N and START must be XE + (multiple of W) with XE=%" PRIu64 "\n", XE); return 1; }
    small_primes(); make_inst();
    if (logname) {                                            // resume: E and pi of the DONE lines, maximum of the REC lines
        FILE *f = fopen(logname, "r"); char line[512];
        if (!f && errno != ENOENT) { fprintf(stderr, "cannot open %s (%s)\n", logname, strerror(errno)); return 1; }
        if (!f) fprintf(stderr, "LOG %s not found: starting from scratch\n", logname);
        if (f) {
            // use only lines that end with '\n' and parse completely (drop partially written lines)
            while (fgets(line, sizeof line, f)) {
                size_t len = strlen(line);
                if (len == 0 || line[len - 1] != '\n') continue;
                while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ' || line[len - 1] == '\t')) line[--len] = 0;   // also tolerates CRLF
                uint64_t a, b, c, d; unsigned v, m1, m2; char t; long u1, u2, an; double wl; int k = -1;
                if (sscanf(line, "REC %c %" SCNu64 " %u%n", &t, &a, &v, &k) == 3 && k == (int)len) { if (t == 'A' && v > recA) recA = v; if (t == 'B' && v > recB) recB = v; }
                else if (k = -1, sscanf(line, "DONE X=%" SCNu64 " E=%" SCNu64 " pi=%" SCNu64 " piseg=%" SCNu64 " uncB=%ld uncA=%ld maxB=%u maxA=%u anom=%ld wall=%lf%n",
                                &a, &b, &c, &d, &u1, &u2, &m1, &m2, &an, &wl, &k) == 10 && k == (int)len) { X0 = b; pi_cum = c; if (an) fail = 1; }
                else if (k = -1, sscanf(line, "EXACT XE=%" SCNu64 " pi=%" SCNu64 " recA=%" SCNu64 " recB=%" SCNu64 "%n", &a, &b, &c, &d, &k) == 4 && k == (int)len) {
                    X0 = a; pi_cum = b; if (c > recA) recA = c; if (d > recB) recB = d; }
                else if (!strncmp(line, "ANOMALY", 7) || !strncmp(line, "CHECK FAIL", 10) || !strncmp(line, "SELFCHECK FAIL", 14) || !strncmp(line, "SIEVE FAIL", 10))
                    fail = 1;   // fail carries over across a resume
            }
            fclose(f);
            if (X0 && (X0 < XE || (X0 - XE) % W)) { fprintf(stderr, "resume point X=%" PRIu64 " from LOG is invalid (W=%" PRIu64 ")\n", X0, W); return 1; }
        }
        logf = fopen(logname, "a"); if (!logf) { perror(logname); return 1; }
    }
    if (X0 == 0) {
        exact_phase(&recA, &recB, &pi_cum); X0 = XE;
        out("EXACT XE=%" PRIu64 " pi=%" PRIu64 " recA=%" PRIu64 " recB=%" PRIu64 "\n", XE, pi_cum, recA, recB);
    }
    if (X0 < XE) { fprintf(stderr, "START must be >= %" PRIu64 "\n", XE); return 1; }
    const uint32_t thr[3] = {SP[K - 1], SP1[K - 1], SP3[K - 1]};
    if (!EXACT && (recB < thr[0] || recA < thr[1] || recA < thr[2])) {
        fprintf(stderr, "premise violated: recB=%" PRIu64 " recA=%" PRIu64 " but K-th primes are %u / %u / %u (give RECA= RECB= or use EXACT=1)\n", recB, recA, thr[0], thr[1], thr[2]);
        return 1;
    }
    out("START X=%" PRIu64 " N=%" PRIu64 " W=%" PRIu64 " SEG=%d K=%d thr=%u/%u/%u recA=%" PRIu64 " recB=%" PRIu64 " threads=%d\n",
        X0, N, W, SEG, K, thr[0], thr[1], thr[2], recA, recB, omp_get_max_threads());
    for (uint64_t S0 = X0; S0 < N; S0 += (uint64_t)SEG * W) {
        const int nwin = (int)((N - S0) / W < (uint64_t)SEG ? (N - S0) / W : (uint64_t)SEG);
        chain_t *chA = must(calloc(nwin, sizeof *chA), "chains"), *chB = must(calloc(nwin, sizeof *chB), "chains"); wstat_t *st = must(calloc(nwin, sizeof *st), "stats");
        const double t0 = omp_get_wtime();
#pragma omp parallel
        {
            primesieve_iterator it; primesieve_init(&it);
            uint64_t *P = must(aligned_alloc(64, ((W / 2 + MARG) / 64 + 16) * 8 / 64 * 64 + 64), "window");   // size is a multiple of 64
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < nwin; i++) do_window(S0 + (uint64_t)i * W, &it, P, &chB[i], &chA[i], &st[i]);
            primesieve_free_iterator(&it); free(P);
        }
        uint64_t piseg = 0; long unc[3] = {0, 0, 0}, anom = 0; uint32_t mx[3] = {0, 0, 0};
        for (int i = 0; i < nwin; i++) {
            if (DEBUG) {
                out("WIN X=%" PRIu64 " cnt=%" PRIu64 " chainB:", S0 + (uint64_t)i * W, st[i].cnt);
                for (int k = 0; k < chB[i].n; k++) out(" %" PRIu64 ":%u", chB[i].a[k].n, chB[i].a[k].v);
                out(" chainA:"); for (int k = 0; k < chA[i].n; k++) out(" %" PRIu64 ":%u", chA[i].a[k].n, chA[i].a[k].v); out("\n");
            }
            for (int k = 0; k < chB[i].n; k++) if (chB[i].a[k].v > recB) { recB = chB[i].a[k].v; out("REC B %" PRIu64 " %u\n", chB[i].a[k].n, (unsigned)recB); }
            for (int k = 0; k < chA[i].n; k++) if (chA[i].a[k].v > recA) { recA = chA[i].a[k].v; out("REC A %" PRIu64 " %u\n", chA[i].a[k].n, (unsigned)recA); }
            piseg += st[i].cnt; anom += st[i].anom;
            for (int k = 0; k < 3; k++) { unc[k] += st[i].unc[k]; if (st[i].mx[k] > mx[k]) mx[k] = st[i].mx[k]; }
            free(chA[i].a); free(chB[i].a);
        }
        pi_cum += piseg; if (anom) fail = 1;
        const uint64_t E = S0 + (uint64_t)nwin * W;
        out("DONE X=%" PRIu64 " E=%" PRIu64 " pi=%" PRIu64 " piseg=%" PRIu64 " uncB=%ld uncA=%ld maxB=%u maxA=%u anom=%ld wall=%.1f\n",
            S0, E, pi_cum, piseg, unc[0], unc[1] + unc[2], mx[0], mx[1] > mx[2] ? mx[1] : mx[2], anom, omp_get_wtime() - t0);
        free(chA); free(chB); free(st);
    }
    out("ALLDONE N=%" PRIu64 " pi=%" PRIu64 " recA=%" PRIu64 " recB=%" PRIu64 " fail=%d\n", N, pi_cum, recA, recB, fail);
    return fail ? 2 : 0;
}
