#pragma once

// ============================================================================
// Factorization.h — Sieve-based Mobius function and prime factorization.
//
// Provides:
//   - fillIsPrime(): Eratosthenes sieve for small primes
//   - fillMuBlock(): compute mu(n)..mu(n+m-1) via trial division
//   - initFactStencil() / fillFactBlock(): factorize a block of integers using
//     a precomputed stencil of period 360360 (primes up to 13)
//   - FacToSumMu(): branchless evaluation of sum-of-Mobius from factorization
//
// All functions are header-only to allow inlining of hot-path code.
// ============================================================================

#include "types.h"
#include <cstdlib>
#include <cstring>

typedef unsigned int uint;
typedef unsigned long ulong;

// ============================================================================
// Prime factorization storage
// ============================================================================

struct PrimFact {
    ulong plist;
    ulong pmark;
};

// Bit-encoding: primes stored as trailing bits after the marker.
// Example: 2,3,5,7 encoded as pmark=111010, plist=010111.

#define LOG2(X) ((unsigned)(8*sizeof(ulong) - 1 - __builtin_clzl((X))))

// ============================================================================
// Sieve of Eratosthenes
// ============================================================================

static inline void fillIsPrime(short* isprime, long N) {
    for (long i = 2; i < N; i++)
        isprime[i] = 1;
    for (long i = 2; i * i < N; i++)
        if (isprime[i])
            for (long j = i; j * i < N; j++)
                isprime[j * i] = 0;
}

// ============================================================================
// Mobius function sieve over a block
// ============================================================================

static inline void fillMuBlock(Int8* mun, short* isprime, long n, long m) {
    long* tabla = (long*)malloc(m * sizeof(long));
    for (long i = 0; i < m; i++)
        tabla[i] = 1;

    for (long i = 0; i < m; i++) {
        if ((i + n) % 2)
            mun[i] = 1;
        else if ((i + n) % 4) {
            mun[i] = -1;
            tabla[i] = 2;
        } else {
            mun[i] = 0;
        }
    }

    for (long p = 3; p * p <= n + m; p++)
        if (isprime[p]) {
            long red = (n % p);
            for (long j = (red ? p - red : 0); j < m; j += p) {
                if (((j + n) / p) % p) {
                    mun[j] = -mun[j];
                    tabla[j] *= p;
                } else {
                    mun[j] = 0;
                }
            }
        }

    for (long i = 0; i < m; i++)
        if (mun[i] != 0 && tabla[i] != n + i)
            mun[i] = -mun[i];

    free(tabla);
}

// ============================================================================
// Precomputed factorization stencil (period 360360 = 2^3 * 3^2 * 5 * 7 * 11 * 13)
// ============================================================================

static constexpr ulong FACT_P = 2UL * 2 * 2 * 3 * 3 * 5 * 7 * 11 * 13;  // 360360
static constexpr ulong FACT_PMAX = 13;

static PrimFact* gStencilFun = nullptr;
static ulong* gStencilSqfprod = nullptr;
static ulong* gStencilTabla = nullptr;

static inline void initFactStencil(short* isprime) {
    if (gStencilFun) return;

    gStencilFun = (PrimFact*)malloc(FACT_P * sizeof(PrimFact));
    gStencilSqfprod = (ulong*)malloc(FACT_P * sizeof(ulong));
    gStencilTabla = (ulong*)malloc(FACT_P * sizeof(ulong));

    for (ulong j = 0; j < FACT_P; j++) {
        gStencilTabla[j] = 1;
        gStencilSqfprod[j] = 1;
    }
    memset(gStencilFun, 0, sizeof(PrimFact) * FACT_P);

    for (ulong p = 2; p <= FACT_PMAX; p++)
        if (isprime[p]) {
            uint pk = LOG2(p);
            ulong nmark = 1ul << (pk - 1);
            ulong trailp = (p & ~((~0ul) << pk));
            for (uint k = 1, d = p; !(FACT_P % d); k++, d *= p) {
                for (ulong j = 0; j < FACT_P; j += d) {
                    if (k == 1) {
                        gStencilFun[j].pmark <<= pk;
                        gStencilFun[j].pmark |= nmark;
                        gStencilFun[j].plist <<= pk;
                        gStencilFun[j].plist |= trailp;
                        gStencilSqfprod[j] *= p;
                    }
                    gStencilTabla[j] *= p;
                }
            }
        }
}

// ============================================================================
// Block factorization with stencil + large-prime sieving
// ============================================================================

static inline void fillFactBlock(
    PrimFact* fun, ulong* sqfprod,
    short* isprime, long n, long m,
    ulong* tabla  // pre-allocated buffer, size >= m
) {
    // Copy precomputed stencil with rotation for offset n
    {
        ulong off = (ulong)(n % FACT_P);
        long i = 0;
        while (i < m) {
            ulong src = (off + i) % FACT_P;
            long chunk = (long)(FACT_P - src);
            if (chunk > m - i) chunk = m - i;
            memcpy(tabla + i,   gStencilTabla + src,   chunk * sizeof(ulong));
            memcpy(sqfprod + i, gStencilSqfprod + src, chunk * sizeof(ulong));
            memcpy(fun + i,     gStencilFun + src,     chunk * sizeof(PrimFact));
            i += chunk;
        }
    }

    // Complete higher powers of small primes beyond what divides FACT_P
    for (ulong p = 2; p <= FACT_PMAX; p++)
        if (isprime[p]) {
            ulong dold, d;
            for (dold = 1, d = p; !(FACT_P % d); dold = d, d *= p);
            for (ulong roof = (n + m - 1) / p; dold <= roof; dold = d, d *= p) {
                long j0 = (n == 0) ? d : ((n + d - 1) / d) * d - n;
                for (long j = j0; j < m; j += d)
                    tabla[j] *= p;
            }
        }

    // Larger primes: split into small (L1-blocked) and large (bucket-scheduled)
    {
#ifndef MHT_FACT_TILE
#define MHT_FACT_TILE 4096
#endif
        static constexpr long BLOCK = MHT_FACT_TILE;
        if (m <= BLOCK) {
            // Small block: single-pass (all primes)
            for (ulong p = FACT_PMAX + 2; p * p <= (ulong)(n + m - 1); p += 2)
                if (isprime[p]) {
                    uint pk = LOG2(p);
                    ulong nmark = 1ul << (pk - 1);
                    ulong trailp = (p & ~((~0ul) << pk));
                    long j0 = (n == 0) ? p : ((n + p - 1) / p) * p - n;
                    for (long j = j0; j < m; j += p) {
                        fun[j].pmark <<= pk;  fun[j].pmark |= nmark;
                        fun[j].plist <<= pk;  fun[j].plist |= trailp;
                        sqfprod[j] *= p;
                        tabla[j] *= p;
                    }
                    // Guard before multiplying: for p > 2^{64/3}, d *= p wraps
                    // mod 2^64 and a wrapped value can land back below n+m-1,
                    // stamping spurious factors (this corrupted M(1e23)).
                    for (ulong d = p * p, roof = (ulong)(n + m - 1); d <= roof; ) {
                        j0 = (n == 0) ? d : ((n + d - 1) / d) * d - n;
                        for (long j = j0; j < m; j += d)
                            tabla[j] *= p;
                        if (d > roof / p) break;
                        d *= p;
                    }
                }
        } else {
            // Large block: collect primes into arrays, split small vs large.
            long nprimes = 0;
            for (ulong p = FACT_PMAX + 2; p * p <= (ulong)(n + m - 1); p += 2)
                if (isprime[p]) nprimes++;

            ulong* pv  = (ulong*)malloc(nprimes * sizeof(ulong));
            ulong* pkv = (ulong*)malloc(nprimes * sizeof(ulong));
            ulong* nmv = (ulong*)malloc(nprimes * sizeof(ulong));
            ulong* tpv = (ulong*)malloc(nprimes * sizeof(ulong));
            long*  nxt = (long*)malloc(nprimes * sizeof(long));

            long pi = 0, nSmall = 0;
            for (ulong p = FACT_PMAX + 2; p * p <= (ulong)(n + m - 1); p += 2)
                if (isprime[p]) {
                    pv[pi] = p;
                    pkv[pi] = LOG2(p);
                    nmv[pi] = 1ul << (pkv[pi] - 1);
                    tpv[pi] = (p & ~((~0ul) << pkv[pi]));
                    nxt[pi] = (n == 0) ? p : ((n + p - 1) / p) * p - n;
                    pi++;
                    if (p <= (ulong)BLOCK) nSmall = pi;
                }

            // L1-blocked sieve for small primes (p <= BLOCK).
            for (long blk = 0; blk < m; blk += BLOCK) {
                long blkEnd = blk + BLOCK;
                if (blkEnd > m) blkEnd = m;
                for (pi = 0; pi < nSmall; pi++) {
                    long j;
                    for (j = nxt[pi]; j < blkEnd; j += pv[pi]) {
                        fun[j].pmark <<= pkv[pi];  fun[j].pmark |= nmv[pi];
                        fun[j].plist <<= pkv[pi];  fun[j].plist |= tpv[pi];
                        sqfprod[j] *= pv[pi];
                        tabla[j] *= pv[pi];
                    }
                    nxt[pi] = j;
                }
            }

            // Bucket-scheduled sieve for large primes (p > BLOCK).
            {
                long nLarge = nprimes - nSmall;
                long nblocks = (m + BLOCK - 1) / BLOCK;

                long* bhead = (long*)malloc(nblocks * sizeof(long));
                long* bnext = (long*)malloc(nLarge * sizeof(long));
                long* bhit  = (long*)malloc(nLarge * sizeof(long));

                for (long b = 0; b < nblocks; b++) bhead[b] = -1;

                for (long i = 0; i < nLarge; i++) {
                    bhit[i] = nxt[nSmall + i];
                    if (bhit[i] < m) {
                        long b = bhit[i] / BLOCK;
                        bnext[i] = bhead[b];
                        bhead[b] = i;
                    }
                }

                for (long blk = 0; blk < nblocks; blk++) {
                    long i = bhead[blk];
                    while (i >= 0) {
                        long nextI = bnext[i];
                        long li = nSmall + i;
                        long j = bhit[i];

                        fun[j].pmark <<= pkv[li];  fun[j].pmark |= nmv[li];
                        fun[j].plist <<= pkv[li];  fun[j].plist |= tpv[li];
                        sqfprod[j] *= pv[li];
                        tabla[j] *= pv[li];

                        j += pv[li];
                        if (j < m) {
                            bhit[i] = j;
                            long nb = j / BLOCK;
                            bnext[i] = bhead[nb];
                            bhead[nb] = i;
                        }

                        i = nextI;
                    }
                }

                free(bhead); free(bnext); free(bhit);
            }

            // Higher powers: single pass (large stride, no blocking benefit)
            // Guard before multiplying: for p > 2^{64/3}, d *= p wraps mod 2^64
            // and a wrapped value can land back below n+m-1, stamping spurious
            // factors (this corrupted M(1e23)).
            for (pi = 0; pi < nprimes; pi++) {
                ulong p = pv[pi];
                for (ulong d = p * p, roof = (ulong)(n + m - 1); d <= roof; ) {
                    long j0 = (n == 0) ? d : ((n + d - 1) / d) * d - n;
                    for (long j = j0; j < m; j += d)
                        tabla[j] *= p;
                    if (d > roof / p) break;
                    d *= p;
                }
            }

            free(pv); free(pkv); free(nmv); free(tpv); free(nxt);
        }
    }

    // Finalize: entries with a large prime factor not found by sieve
    for (long j = 0; j < m; j++)
        if (tabla[j] != (ulong)(n + j)) {
            ulong p = (n + j) / tabla[j];
            uint pk = LOG2(p);
            fun[j].pmark = ((fun[j].pmark << 1) | 1ul) << (pk - 1);
            fun[j].plist = (fun[j].plist << pk) | (p & ~((~0ul) << pk));
            sqfprod[j] *= p;
        }
}

// ============================================================================
// FacToSumMu — evaluate sum-of-Mobius from prime factorization
// ============================================================================

static inline long subFacToSumMu(PrimFact f, ulong m, ulong mc, long a, ulong n) {
    if (m > (ulong)a) return 0;
    if (!f.pmark) return 1;
    if (mc >= n) return 0;

    uint k = __builtin_ctzl(f.pmark) + 1;
    f.pmark >>= k;
    ulong p = (1ul << k) | (f.plist & ~((~0ul) << k));
    f.plist >>= k;

    return subFacToSumMu(f, m, mc * p, a, n)
         - subFacToSumMu(f, m * p, mc, a, n);
}

static inline ulong extractPrime(ulong* pmark, ulong* plist) {
    uint k = __builtin_ctzl(*pmark) + 1;
    *pmark >>= k;
    ulong p = (1ul << k) | (*plist & ~((~0ul) << k));
    *plist >>= k;
    return p;
}

// Branchless helper for 2 remaining primes.
static inline long sub2Primes(ulong m, ulong mc, ulong p4, ulong p5, ulong ua, ulong nsqf) {
    long c1 = (long)(mc * p4 < nsqf) & (long)(m * p5 > ua);
    long c2 = (long)(m * p4 <= ua) & (long)(m * p4 * p5 > ua);
    return c1 - c2;
}

static inline long FacToSumMu(PrimFact f, long a, ulong nsqf) {
    ulong pmark = f.pmark, plist = f.plist;
    ulong ua = (ulong)a;

    if (!pmark) return 1;
    if (ua >= nsqf) return 0;

    ulong p1 = extractPrime(&pmark, &plist);

    if (!pmark) {
        // nprimes == 1: branchless
        return (long)(p1 > ua);
    }

    ulong p2 = extractPrime(&pmark, &plist);

    if (!pmark) {
        // nprimes == 2: branchless
        long c_ap1 = (long)(a * p1 < nsqf);
        long c_p1  = (long)(p1 <= ua);
        return (c_ap1 - (c_ap1 & (long)(p2 <= ua)))
             - (c_p1  - (c_p1  & (long)(p1 * p2 <= ua)));
    }

    ulong p3 = extractPrime(&pmark, &plist);

    if (!pmark) {
        // nprimes == 3: branchless
        ulong ap1 = a * p1, ap1p2 = ap1 * p2, ap2 = a * p2;
        long c_ap1   = (long)(ap1 < nsqf);
        long c_ap1p2 = c_ap1 & (long)(ap1p2 < nsqf);
        long c_p2    = c_ap1 & (long)(p2 <= ua);
        long c_p1    = (long)(p1 <= ua);
        long c_ap2   = c_p1 & (long)(ap2 < nsqf);
        long c_p1p2  = c_p1 & (long)(p1 * p2 <= ua);
        return (c_ap1p2 - (c_ap1p2 & (long)(p3 <= ua)))
             - (c_p2    - (c_p2    & (long)(p2 * p3 <= ua)))
             - (c_ap2   - (c_ap2   & (long)(p1 * p3 <= ua)))
             + (c_p1p2  - (c_p1p2  & (long)(p1 * p2 * p3 <= ua)));
    }

    ulong p4 = extractPrime(&pmark, &plist);

    if (!pmark) {
        // nprimes == 4: branchless
        ulong ap1 = a * p1, ap1p2 = ap1 * p2, ap2 = a * p2;
        long c_ap1   = (long)(ap1 < nsqf);
        long c_ap1p2 = c_ap1 & (long)(ap1p2 < nsqf);
        long c_p2    = c_ap1 & (long)(p2 <= ua);
        long c_p1    = (long)(p1 <= ua);
        long c_ap2   = c_p1 & (long)(ap2 < nsqf);
        long c_p1p2  = c_p1 & (long)(p1 * p2 <= ua);
        return (c_ap1p2 & (long)(ap1p2 * p3 < nsqf) & (long)(p4 > ua))
             - (c_ap1p2 & (long)(p3 <= ua) & (long)(p3 * p4 > ua))
             - (c_p2    & (long)(ap1 * p3 < nsqf) & (long)(p2 * p4 > ua))
             + (c_p2    & (long)(p2 * p3 <= ua) & (long)(p2 * p3 * p4 > ua))
             - (c_ap2   & (long)(ap2 * p3 < nsqf) & (long)(p1 * p4 > ua))
             + (c_ap2   & (long)(p1 * p3 <= ua) & (long)(p1 * p3 * p4 > ua))
             + (c_p1p2  & (long)(a * p3 < nsqf) & (long)(p1 * p2 * p4 > ua))
             - (c_p1p2  & (long)(p1 * p2 * p3 <= ua) & (long)(p1 * p2 * p3 * p4 > ua));
    }

    ulong p5 = extractPrime(&pmark, &plist);

    if (!pmark) {
        // nprimes == 5: sub2Primes for the final 2 primes
        long result = 0;
        ulong ap1 = a * p1;
        if (ap1 < nsqf) {
            ulong ap1p2 = ap1 * p2;
            if (ap1p2 < nsqf) {
                if (ap1p2 * p3 < nsqf)
                    result += sub2Primes(1, ap1p2 * p3, p4, p5, ua, nsqf);
                if (p3 <= ua)
                    result -= sub2Primes(p3, ap1p2, p4, p5, ua, nsqf);
            }
            if (p2 <= ua) {
                if (ap1 * p3 < nsqf)
                    result -= sub2Primes(p2, ap1 * p3, p4, p5, ua, nsqf);
                if (p2 * p3 <= ua)
                    result += sub2Primes(p2 * p3, ap1, p4, p5, ua, nsqf);
            }
        }
        if (p1 <= ua) {
            ulong ap2 = a * p2;
            if (ap2 < nsqf) {
                if (ap2 * p3 < nsqf)
                    result -= sub2Primes(p1, ap2 * p3, p4, p5, ua, nsqf);
                if (p1 * p3 <= ua)
                    result += sub2Primes(p1 * p3, ap2, p4, p5, ua, nsqf);
            }
            if (p1 * p2 <= ua) {
                ulong ap3 = a * p3;
                if (ap3 < nsqf)
                    result += sub2Primes(p1 * p2, ap3, p4, p5, ua, nsqf);
                if (p1 * p2 * p3 <= ua)
                    result -= sub2Primes(p1 * p2 * p3, a, p4, p5, ua, nsqf);
            }
        }
        return result;
    }

    // nprimes >= 6: unroll first 3 levels, recurse for remainder
    PrimFact frest = f;
    ulong rm = frest.pmark, rl = frest.plist;
    extractPrime(&rm, &rl);
    extractPrime(&rm, &rl);
    extractPrime(&rm, &rl);
    frest.pmark = rm;
    frest.plist = rl;

    long result = 0;
    ulong ap1 = a * p1;
    if (ap1 < nsqf) {
        ulong ap1p2 = ap1 * p2;
        if (ap1p2 < nsqf) {
            if (ap1p2 * p3 < nsqf)
                result += subFacToSumMu(frest, 1, ap1p2 * p3, a, nsqf);
            if (p3 <= ua)
                result -= subFacToSumMu(frest, p3, ap1p2, a, nsqf);
        }
        if (p2 <= ua) {
            if (ap1 * p3 < nsqf)
                result -= subFacToSumMu(frest, p2, ap1 * p3, a, nsqf);
            if (p2 * p3 <= ua)
                result += subFacToSumMu(frest, p2 * p3, ap1, a, nsqf);
        }
    }
    if (p1 <= ua) {
        ulong ap2 = a * p2;
        if (ap2 < nsqf) {
            if (ap2 * p3 < nsqf)
                result -= subFacToSumMu(frest, p1, ap2 * p3, a, nsqf);
            if (p1 * p3 <= ua)
                result += subFacToSumMu(frest, p1 * p3, ap2, a, nsqf);
        }
        if (p1 * p2 <= ua) {
            ulong ap3 = a * p3;
            if (ap3 < nsqf)
                result += subFacToSumMu(frest, p1 * p2, ap3, a, nsqf);
            if (p1 * p2 * p3 <= ua)
                result -= subFacToSumMu(frest, p1 * p2 * p3, a, a, nsqf);
        }
    }
    return result;
}
