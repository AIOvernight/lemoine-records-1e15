// lemgpu.cu — GPU exhaustive check of Lemoine's conjecture (odd n = 2p + q, p,q prime) and extension of the two
// record sequences, on the same window grid and with the same statistics as yygotm/lemoine-records-1e15 (lemoine.c):
//   B: least p (doubled term) records — OEIS A195353/A195354.   n in (X, X+W)
//   A: least q (undoubled term) records — OEIS A194828/A194829. n in (2X, 2X+2W)
// Window: X = XE + k*W, XE = 2^28, W = 2^30. Bitmap P of odd numbers in [lo, X+W), lo = X - 2*MARG + 1 (bit j <-> lo + 2j), 1 = prime.
// GPU: (1) segmented sieve of Eratosthenes into P (shared-memory blocks for primes < 2^20, global atomics for larger primes),
//      (2) cover kernel: OR of the first K shifted copies of P (per instance), residual scan for uncovered n, stats + record candidates.
// CPU: optional SELFCHECK (bit-exact comparison of P against a primesieve-built bitmap for sampled windows).
// Build: nvcc -O3 -arch=sm_120 -Xcompiler -fopenmp lemgpu.cu -o lemgpu -lprimesieve
// Usage: lemgpu END [START=X0] [RECA=q] [RECB=p] [SEG=240] [K=128] [LOG=file] [SELFCHECK=k] [NAIVE=k] [SUBBITS=20] [PICUM=pi]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <cinttypes>
#include <vector>
#include <algorithm>
#include <primesieve.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MARG 16384L
#define XE ((uint64_t)1 << 28)
static const uint64_t W = 1ULL << 30;
static int K = 128, SEG = 240, SELFCHECK = 0, NAIVE = 0, SUBBITS = 20;
static FILE *g_logf;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(9); } } while (0)

static void out(const char *fmt, ...) {
    va_list ap; char buf[1024];
    va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    fputs(buf, stdout); fflush(stdout);
    if (g_logf) { fputs(buf, g_logf); fflush(g_logf); }
}

// ---------------- small primes & shift tables (identical conventions to lemoine.c) ----------------
static std::vector<uint32_t> SP, SP1, SP3;
static void small_primes() {
    long lim = 4 * MARG + 4;
    std::vector<uint8_t> c(lim + 1, 0);
    for (long i = 2; i * i <= lim; i++) if (!c[i]) for (long j = i * i; j <= lim; j += i) c[j] = 1;
    for (long i = 2; i <= lim; i++) if (!c[i]) { SP.push_back(i); if (i % 4 == 1) SP1.push_back(i); else if (i % 4 == 3) SP3.push_back(i); }
}
#define MAXS 4096
__constant__ int cS[3][MAXS];      // shifts: idx into P = t + S[j]
__constant__ int cNS[3];
__constant__ int cOff[3][256];   // word offsets of the first K shifts, sorted by bit offset (S & 31)
__constant__ int cSh[3][256];    // bit offsets, same order
static int hNS[3]; static std::vector<int> hS[3], hPR[3];
static void make_inst() {
    const std::vector<uint32_t> *pr[3] = {&SP, &SP1, &SP3};
    for (int k = 0; k < 3; k++) {
        for (size_t j = 0; j < pr[k]->size(); j++) {
            uint32_t p = (*pr[k])[j];
            long s = k == 0 ? MARG - (long)p : k == 1 ? MARG - (long)(p - 1) / 4 : MARG - 1 - (long)(p - 3) / 4;
            if (s < 0) break;
            hS[k].push_back((int)s); hPR[k].push_back((int)p);
        }
        hNS[k] = (int)hS[k].size();
        if (hNS[k] > MAXS || hNS[k] < K) { fprintf(stderr, "bad nS[%d]=%d\n", k, hNS[k]); exit(1); }
        CK(cudaMemcpyToSymbol(cS, hS[k].data(), hNS[k] * 4, k * MAXS * 4));
    }
    CK(cudaMemcpyToSymbol(cNS, hNS, 12));
    for (int k = 0; k < 3; k++) {
        std::vector<std::pair<int,int>> v; for (int j = 0; j < K; j++) v.push_back({hS[k][j] >> 5, hS[k][j] & 31});   // (word offset, bit offset)
        std::sort(v.begin(), v.end());
        int off[256], shv[256];
        for (int j = 0; j < K; j++) { off[j] = v[j].first; shv[j] = v[j].second; }
        CK(cudaMemcpyToSymbol(cOff, off, K * 4, k * 256 * 4)); CK(cudaMemcpyToSymbol(cSh, shv, K * 4, k * 256 * 4));
    }
}

// ---------------- GPU sieve ----------------
// wheel: primes 3..31 handled by precomputed 32-bit masks instead of atomics. cWM[p-index][r] = bits b in [0,32) with b = r (mod p)
#define NWP 10
__constant__ uint32_t cWM[NWP][32];
__constant__ int cWP[NWP];
static const int hWP[NWP] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
static void make_wheel() {
    uint32_t wm[NWP][32]; memset(wm, 0, sizeof wm);
    for (int i = 0; i < NWP; i++) for (int r = 0; r < hWP[i]; r++) for (int b = r; b < 32; b += hWP[i]) wm[i][r] |= 1u << b;
    CK(cudaMemcpyToSymbol(cWM, wm, sizeof wm)); CK(cudaMemcpyToSymbol(cWP, hWP, sizeof hWP));
}
struct WheelOff { int j0[NWP]; };   // j0[i]: bit index (global, in P) of the first odd multiple of prime i
// P32: bitmap as 32-bit words, 1 = prime (odd numbers), L bits valid.
// k_small: each block sieves SUB = 2^SUBBITS bits in shared memory with all primes p < pmaxSmall (p >= 3).
__global__ void __launch_bounds__(1024, 2) k_small(uint32_t *P32, long L, uint64_t lo, const uint32_t *pr, int npr, int subbits, WheelOff wo, int usewheel) {
    extern __shared__ uint32_t sh[];
    __shared__ uint32_t swm[NWP][32];
    const long SUB = 1L << subbits, nw = SUB >> 5;
    const long b0 = (long)blockIdx.x * SUB;                       // first bit of this block
    for (long w = threadIdx.x; w < nw; w += blockDim.x) sh[w] = 0;
    if (threadIdx.x < NWP * 32) (&swm[0][0])[threadIdx.x] = (&cWM[0][0])[threadIdx.x];
    __syncthreads();
    // primes with many marks per block: cooperative marking (block-wide loop over primes, threads stride the multiples)
    int i = 0;
    for (; i < npr; i++) {
        const uint32_t p = pr[i];
        if ((long)p > (SUB >> 8)) break;                              // > 256 marks per block → cooperative
        // first odd multiple of p >= lo + 2*b0
        const uint64_t a = lo + 2 * (uint64_t)b0;
        uint64_t m = a % p; uint64_t f = m ? a + (p - m) : a;
        if (!(f & 1)) f += p;
        long j = (long)((f - a) >> 1);                                 // bit within block
        for (long jj = j + (long)threadIdx.x * p; jj < SUB; jj += (long)blockDim.x * p) atomicOr(&sh[jj >> 5], 1u << (jj & 31));
    }
    __syncthreads();
    for (int q = i + threadIdx.x; q < npr; q += blockDim.x) {
        const uint32_t p = pr[q];
        const uint64_t a = lo + 2 * (uint64_t)b0;
        uint64_t m = a % p; uint64_t f = m ? a + (p - m) : a;
        if (!(f & 1)) f += p;
        for (long jj = (long)((f - a) >> 1); jj < SUB; jj += p) atomicOr(&sh[jj >> 5], 1u << (jj & 31));
    }
    __syncthreads();
    int bp[NWP];
#pragma unroll
    for (int i = 0; i < NWP; i++) bp[i] = (int)(b0 % cWP[i]);
    for (long w = threadIdx.x; w < nw; w += blockDim.x) {
        const long gb = b0 + w * 32;
        if (gb >= L) break;
        const int w32 = (int)w;
        uint32_t comp = sh[w];
        if (usewheel)
#pragma unroll
        for (int i = 0; i < NWP; i++) { const int p = cWP[i]; int r = (wo.j0[i] - bp[i] - ((w32 * 32) % p)) % p; if (r < 0) r += p; comp |= swm[i][r]; }
        uint32_t v = ~comp;
        if (gb + 32 > L) v &= (1u << (L - gb)) - 1u;
        P32[(b0 >> 5) + w] = v;
    }
}
// k_large: thread per prime p >= pmaxSmall; clears composite bits with global atomics.
__global__ void k_large(uint32_t *P32, long L, uint64_t lo, const uint32_t *pr, int npr) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= npr) return;
    const uint32_t p = pr[q];
    uint64_t m = lo % p; uint64_t f = m ? lo + (p - m) : lo;
    if (!(f & 1)) f += p;
    // f = mm*p with mm odd; only mm coprime to 3 and 5 need marking (others are marked by the wheel). residues mod 30: 1,7,11,13,17,19,23,29
    const int gaps[8] = {6, 4, 2, 4, 2, 4, 6, 2};
    uint64_t mm = f / p; int r = (int)(mm % 30);
    static const int cyc[30] = {0,0,1,1,1,1,1,1,2,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6,7,7,7,7,7,7};   // index of the next allowed residue >= r (for odd r)
    static const int res[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    int c = cyc[r]; int adv = res[c] - r; if (adv < 0) adv += 30;   // r=29 -> c=7 adv=0
    long jj = (long)((f - lo) >> 1) + (long)(adv / 2) * p;
    while (jj < L) { atomicAnd(&P32[jj >> 5], ~(1u << (jj & 31))); jj += (long)(gaps[c] / 2) * p; c = (c + 1) & 7; }
}
// popcount of bits [b0, L)
__global__ void k_count(const uint32_t *P32, long L, long b0, unsigned long long *cnt) {
    const long w0 = b0 >> 5, w1 = (L + 31) >> 5;
    unsigned long long c = 0;
    for (long w = w0 + blockIdx.x * blockDim.x + threadIdx.x; w < w1; w += (long)gridDim.x * blockDim.x) {
        uint32_t v = P32[w];
        if (w == w0 && (b0 & 31)) v &= ~((1u << (b0 & 31)) - 1u);
        if (w == w1 - 1 && (L & 31)) v &= (1u << (L & 31)) - 1u;
        c += __popc(v);
    }
    // warp reduce
    for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(cnt, c);
}

// ---------------- cover kernel ----------------
struct Stats { unsigned long long unc[3]; unsigned mx[3]; unsigned anom; unsigned ncand; unsigned long long cnt; unsigned nunc; unsigned pad; };
struct Cand { unsigned t; unsigned v; unsigned kind; unsigned pad; };
#define MAXCAND 262144
#define MAXUNC (1u << 23)
#define CB 256                       // threads (32-bit output words) per cover block
#define TILE (CB + MARG / 32 + 8)    // words of P needed by one block
__global__ void __launch_bounds__(CB) k_cover(const uint64_t *P, long nwords, int kind, int K, unsigned thr, Stats *st, Cand *cand, unsigned *unc_list) {
    __shared__ uint32_t tile[TILE];
    const long w = (long)blockIdx.x * CB + threadIdx.x;   // 32-bit output word index
    const bool active = w < nwords;
    const uint32_t *P32 = (const uint32_t *)P;
    const long wb = (long)blockIdx.x * CB;
    for (int i = threadIdx.x; i < TILE; i += CB) tile[i] = __ldg(&P32[wb + i]);
    __syncthreads();
    const long base = active ? w * 32 : 0;
    const int tid = threadIdx.x;
    uint32_t c = 0;
    int co = -1; uint32_t a = 0, b = 0;
    for (int j = 0; j < K; j++) {
        const int o = cOff[kind][j], sh = cSh[kind][j];
        if (o != co) { a = tile[tid + o]; b = tile[tid + o + 1]; co = o; }
        c |= sh ? __funnelshift_r(a, b, sh) : a;
    }
    uint32_t u = active ? ~c : 0u;
    // warp-aggregated append of uncovered t's to the list
    const int cnt = __popc(u);
    const unsigned FULL = 0xffffffffu; const int lane = threadIdx.x & 31;
    int pre = cnt;
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(FULL, pre, o); if (lane >= o) pre += t; }
    const int tot = __shfl_sync(FULL, pre, 31);
    unsigned wbase = 0;
    if (lane == 31 && tot) wbase = atomicAdd(&st->nunc, (unsigned)tot);
    wbase = __shfl_sync(FULL, wbase, 31);
    unsigned pos = wbase + pre - cnt;
    while (u) { const int bi = __ffs(u) - 1; u &= u - 1; if (pos < MAXUNC) unc_list[pos] = (unsigned)(base + bi); pos++; }
}
__global__ void k_scan(const uint64_t *P, const unsigned *unc_list, int kind, int K, unsigned thr, Stats *st, Cand *cand) {
    const unsigned nunc = min(st->nunc, (unsigned)MAXUNC);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nunc; i += gridDim.x * blockDim.x) {
    const long t = unc_list[i];
    const int nS = cNS[kind]; const int *S = cS[kind];
    unsigned v = 0;
    for (int j = K; j < nS; j++) { const long idx = t + S[j]; if ((P[idx >> 6] >> (idx & 63)) & 1) { const int sv = S[j]; v = kind == 0 ? MARG - sv : kind == 1 ? 4 * (MARG - sv) + 1 : 4 * (MARG - 1 - sv) + 3; break; } }
    if (!v) { atomicAdd(&st->anom, 1u); unsigned c = atomicAdd(&st->ncand, 1u); if (c < MAXCAND) cand[c] = {(unsigned)t, 0u, (unsigned)kind, 0u}; continue; }
    atomicMax(&st->mx[kind], v);
    if (v > thr) { unsigned c = atomicAdd(&st->ncand, 1u); if (c < MAXCAND) cand[c] = {(unsigned)t, v, (unsigned)kind, 0u}; }
    }
}
__global__ void k_fin(Stats *st, int kind) { if (st->nunc > MAXUNC) st->anom += 1000000u; st->unc[kind] = st->nunc; st->nunc = 0; }

static inline uint64_t n_of(int kind, uint64_t X, uint64_t t) {
    return kind == 0 ? X + 1 + 2 * t : kind == 1 ? 2 * X + 4 * t + 3 : 2 * X + 1 + 4 * t;
}

// ---------------- CPU reference bitmap (primesieve) ----------------
static void cpu_bitmap(uint64_t lo, uint64_t hi, uint64_t *P, long LW) {
    memset(P, 0, LW * 8);
    const int nt = omp_get_max_threads();
    const uint64_t span = (hi - lo) / (nt * 2) / 128 * 128 + 128;   // chunk boundaries multiples of 128 numbers = 64 bits → word-aligned
#pragma omp parallel
    {
        primesieve_iterator it; primesieve_init(&it);
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nt * 2; i++) {
            uint64_t a = lo + span * i, b = std::min(hi, a + span);
            if (a >= hi) continue;
            primesieve_jump_to(&it, a, b);
            uint64_t p;
            while ((p = primesieve_next_prime(&it)) < b) { uint64_t j = (p - lo) / 2; P[j >> 6] |= 1ULL << (j & 63); }
        }
        primesieve_free_iterator(&it);
    }
}
// naive per-n check of one window from a bitmap (CPU, independent of the OR trick): recomputes unc/mx for the 3 instances
static void naive_window(const uint64_t *P, unsigned long long unc[3], unsigned mx[3]) {
    const long T = W / 2;
    for (int k = 0; k < 3; k++) {
        unsigned long long u = 0; unsigned m = 0;
#pragma omp parallel for reduction(+:u) reduction(max:m) schedule(static)
        for (long t = 0; t < T; t++) {
            int j = 0;
            for (; j < hNS[k]; j++) { const long idx = t + hS[k][j]; if ((P[idx >> 6] >> (idx & 63)) & 1) break; }
            if (j >= K) { u++; if (j < hNS[k]) { if ((unsigned)hPR[k][j] > m) m = hPR[k][j]; } }
        }
        unc[k] = u; mx[k] = m;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: lemgpu END [START=] [RECA=] [RECB=] [SEG=] [K=] [LOG=] [SELFCHECK=k] [NAIVE=k] [SUBBITS=] [PICUM=]\n"); return 1; }
    uint64_t END = strtoull(argv[1], 0, 0), X0 = XE, recA = 0, recB = 0, pi_cum = 0; const char *logname = 0;
    for (int i = 2; i < argc; i++) {
        if (!strncmp(argv[i], "START=", 6)) X0 = strtoull(argv[i] + 6, 0, 0);
        else if (!strncmp(argv[i], "RECA=", 5)) recA = strtoull(argv[i] + 5, 0, 0);
        else if (!strncmp(argv[i], "RECB=", 5)) recB = strtoull(argv[i] + 5, 0, 0);
        else if (!strncmp(argv[i], "SEG=", 4)) SEG = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "K=", 2)) K = atoi(argv[i] + 2);
        else if (!strncmp(argv[i], "LOG=", 4)) logname = argv[i] + 4;
        else if (!strncmp(argv[i], "SELFCHECK=", 10)) SELFCHECK = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "NAIVE=", 6)) NAIVE = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "SUBBITS=", 8)) SUBBITS = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "PICUM=", 6)) pi_cum = strtoull(argv[i] + 6, 0, 0);
        else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }
    if ((END - XE) % W || (X0 - XE) % W || X0 < XE) { fprintf(stderr, "START/END must be XE + k*W\n"); return 1; }
    if (logname) { g_logf = fopen(logname, "a"); if (!g_logf) { perror(logname); return 1; } }
    if (K > 256) { fprintf(stderr, "K<=256\n"); return 1; }
    small_primes(); make_inst();
    const uint32_t thr[3] = {SP[K - 1], SP1[K - 1], SP3[K - 1]};
    if (recB < thr[0] || recA < thr[1] || recA < thr[2]) { fprintf(stderr, "premise violated: give RECA/RECB >= K-th primes %u/%u/%u\n", thr[0], thr[1], thr[2]); return 1; }
    // sieving primes up to sqrt(END + W)
    uint64_t sq = 1; while (sq * sq < END + W) sq++;
    size_t npr; uint32_t *hpr = (uint32_t *)primesieve_generate_primes(3, sq, &npr, UINT32_PRIMES);
    const uint32_t PSMALL = 1u << 20;
    const int usewheel = getenv("WHEEL") ? atoi(getenv("WHEEL")) : 1;
    int skip = 0; if (usewheel) while (skip < (int)npr && hpr[skip] < 37) skip++;   // wheel primes
    make_wheel();
    int nsmall = 0; while (nsmall < (int)npr && hpr[nsmall] < PSMALL) nsmall++;
    if (X0 - 2 * MARG + 1 <= sq) { fprintf(stderr, "window too low for this sieve (lo <= sqrt)\n"); return 1; }
    uint32_t *dpr; CK(cudaMalloc(&dpr, npr * 4)); CK(cudaMemcpy(dpr, hpr, npr * 4, cudaMemcpyHostToDevice));
    const long L = W / 2 + MARG, LW = L / 64 + 16;
    uint64_t *dP; CK(cudaMalloc(&dP, LW * 8));
    Stats *dst; CK(cudaMalloc(&dst, sizeof(Stats)));
    Cand *dcand; CK(cudaMalloc(&dcand, MAXCAND * sizeof(Cand)));
    unsigned *dunc; CK(cudaMalloc(&dunc, MAXUNC * 4));
    std::vector<Cand> hcand(MAXCAND);
    uint64_t *hP = 0; if (SELFCHECK || NAIVE) hP = (uint64_t *)aligned_alloc(64, LW * 8);
    uint64_t *gP = 0; if (SELFCHECK) gP = (uint64_t *)aligned_alloc(64, LW * 8);
    const long SUB = 1L << SUBBITS; const int nblk = (int)((L + SUB - 1) / SUB);
    CK(cudaFuncSetAttribute(k_small, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SUB / 8)));
    out("START X=%" PRIu64 " END=%" PRIu64 " W=%" PRIu64 " SEG=%d K=%d thr=%u/%u/%u recA=%" PRIu64 " recB=%" PRIu64 " sieve_primes=%zu nsmall=%d SUBBITS=%d gpu=lemgpu\n",
        X0, END, W, SEG, K, thr[0], thr[1], thr[2], recA, recB, npr, nsmall, SUBBITS);
    int fail = 0; long win_index = 0;
    const int TIMING = getenv("TIMING") ? 1 : 0; cudaEvent_t ev[6]; for (int i = 0; i < 6; i++) CK(cudaEventCreate(&ev[i])); double tk[5] = {0,0,0,0,0};
    for (uint64_t S0 = X0; S0 < END; S0 += (uint64_t)SEG * W) {
        const int nwin = (int)std::min<uint64_t>((END - S0) / W, (uint64_t)SEG);
        const double t0 = omp_get_wtime();
        uint64_t piseg = 0; unsigned long long unc[3] = {0, 0, 0}; unsigned mx[3] = {0, 0, 0}; unsigned anom = 0;
        for (int i = 0; i < nwin; i++, win_index++) {
            const uint64_t X = S0 + (uint64_t)i * W, lo = X - 2 * MARG + 1, hi = X + W;
            WheelOff wo;
            for (int i = 0; i < NWP; i++) { const uint64_t p = hWP[i], m = lo % p; uint64_t f = m ? lo + (p - m) : lo; if (!(f & 1)) f += p; wo.j0[i] = (int)((f - lo) >> 1); }
            CK(cudaMemsetAsync(dP, 0, LW * 8));
            CK(cudaMemsetAsync(dst, 0, sizeof(Stats)));
            if (TIMING) cudaEventRecord(ev[0]);
            k_small<<<nblk, 1024, SUB / 8>>>((uint32_t *)dP, L, lo, dpr + skip, nsmall - skip, SUBBITS, wo, usewheel);
            if (TIMING) cudaEventRecord(ev[1]);
            if ((int)npr > nsmall) k_large<<<(int)((npr - nsmall + 255) / 256), 256>>>((uint32_t *)dP, L, lo, dpr + nsmall, (int)(npr - nsmall));
            if (TIMING) cudaEventRecord(ev[2]);
            k_count<<<1024, 256>>>((uint32_t *)dP, L, MARG, &dst->cnt);
            if (TIMING) cudaEventRecord(ev[3]);
            for (int k = 0; k < 3; k++) {
                k_cover<<<(int)((W / 2 / 32 + CB - 1) / CB), CB>>>(dP, W / 2 / 32, k, K, k == 0 ? (unsigned)recB : (unsigned)recA, dst, dcand, dunc);
                k_scan<<<4096, 256>>>(dP, dunc, k, K, k == 0 ? (unsigned)recB : (unsigned)recA, dst, dcand);
                k_fin<<<1, 1>>>(dst, k);
            }
            if (TIMING) cudaEventRecord(ev[4]);
            Stats st; CK(cudaMemcpy(&st, dst, sizeof st, cudaMemcpyDeviceToHost));
            if (TIMING) { cudaEventRecord(ev[5]); cudaEventSynchronize(ev[5]); for (int i = 0; i < 5; i++) { float ms; cudaEventElapsedTime(&ms, ev[i], ev[i+1]); tk[i] += ms; } }
            CK(cudaGetLastError());
            if (st.ncand > MAXCAND) { out("CAND OVERFLOW X=%" PRIu64 " ncand=%u\n", X, st.ncand); fail = 1; st.ncand = MAXCAND; }
            if (st.ncand) CK(cudaMemcpy(hcand.data(), dcand, st.ncand * sizeof(Cand), cudaMemcpyDeviceToHost));
            bool doself = SELFCHECK && (win_index < 2 || win_index % SELFCHECK == 0);
            bool donaive = NAIVE && (win_index < 1 || win_index % NAIVE == 0);
            if (doself || donaive) {
                cpu_bitmap(lo, hi, hP, LW);
                if (doself) {
                    CK(cudaMemcpy(gP, dP, LW * 8, cudaMemcpyDeviceToHost));
                    long bad = 0; for (long w = 0; w < LW; w++) if (gP[w] != hP[w]) bad++;
                    unsigned long long c2 = 0; for (long w = MARG / 64; w < L / 64; w++) c2 += __builtin_popcountll(hP[w]);
                    out("SELFCHECK X=%" PRIu64 " words_diff=%ld cnt_gpu=%llu cnt_cpu=%llu %s\n", X, bad, st.cnt, c2, (bad || c2 != st.cnt) ? "FAIL" : "ok");
                    if (bad || c2 != st.cnt) fail = 1;
                }
                if (donaive) {
                    unsigned long long nu[3]; unsigned nm[3]; naive_window(hP, nu, nm);
                    bool ok = true; for (int k = 0; k < 3; k++) if (nu[k] != st.unc[k] || nm[k] != st.mx[k]) ok = false;
                    out("NAIVE X=%" PRIu64 " gpu unc=%llu/%llu/%llu mx=%u/%u/%u cpu unc=%llu/%llu/%llu mx=%u/%u/%u %s\n", X,
                        st.unc[0], st.unc[1], st.unc[2], st.mx[0], st.mx[1], st.mx[2], nu[0], nu[1], nu[2], nm[0], nm[1], nm[2], ok ? "ok" : "FAIL");
                    if (!ok) fail = 1;
                }
            }
            // records: candidates (v > rec at window start), processed in n order per instance group (B separately from A0+A1)
            std::vector<std::pair<uint64_t, unsigned>> cb, ca;
            for (unsigned c = 0; c < st.ncand; c++) {
                const Cand &cd = hcand[c];
                if (cd.v == 0) { out("ANOMALY %c n=%" PRIu64 " (no prime within margin)\n", "BAA"[cd.kind], n_of(cd.kind, X, cd.t)); fail = 1; continue; }
                (cd.kind == 0 ? cb : ca).push_back({n_of(cd.kind, X, cd.t), cd.v});
            }
            std::sort(cb.begin(), cb.end()); std::sort(ca.begin(), ca.end());
            for (auto &r : cb) if (r.second > recB) { recB = r.second; out("REC B %" PRIu64 " %u\n", r.first, r.second); }
            for (auto &r : ca) if (r.second > recA) { recA = r.second; out("REC A %" PRIu64 " %u\n", r.first, r.second); }
            piseg += st.cnt; anom += st.anom;
            for (int k = 0; k < 3; k++) { unc[k] += st.unc[k]; mx[k] = std::max(mx[k], st.mx[k]); }
        }
        pi_cum += piseg; if (anom) fail = 1;
        const uint64_t E = S0 + (uint64_t)nwin * W; const double dt = omp_get_wtime() - t0;
        if (TIMING) { out("TIMING ms/window: small=%.2f large=%.2f count=%.2f cover=%.2f sync=%.2f\n", tk[0]/nwin, tk[1]/nwin, tk[2]/nwin, tk[3]/nwin, tk[4]/nwin); for (int i = 0; i < 5; i++) tk[i] = 0; }
        out("DONE X=%" PRIu64 " E=%" PRIu64 " pi=%" PRIu64 " piseg=%" PRIu64 " uncB=%llu uncA=%llu maxB=%u maxA=%u anom=%u wall=%.1f rate=%.3g\n",
            S0, E, pi_cum, piseg, unc[0], unc[1] + unc[2], mx[0], std::max(mx[1], mx[2]), anom, dt, (double)(E - S0) / dt);
    }
    out("ALLDONE END=%" PRIu64 " pi=%" PRIu64 " recA=%" PRIu64 " recB=%" PRIu64 " fail=%d\n", END, pi_cum, recA, recB, fail);
    return fail ? 2 : 0;
}
