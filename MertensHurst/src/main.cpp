#include "MertensHurst.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <sys/time.h>

// Parse a nonnegative integer in plain decimal ("2000000") or scientific
// notation ("2e6", "2.5e21"). Returns false if the input is malformed, not
// an exact integer, or too large for UInt128.
static bool parseNumber(const char* s, UInt128& out) {
    UInt128 mant = 0;
    int frac = -1;
    long exp10 = 0;

    if (*s == '\0') return false;
    for (; *s && *s != 'e' && *s != 'E'; ++s) {
        if (*s == '.') {
            if (frac >= 0) return false;
            frac = 0;
        } else if (*s >= '0' && *s <= '9') {
            if (mant > ((UInt128)-1 - 9) / 10) return false;
            mant = mant * 10 + (*s - '0');
            if (frac >= 0) ++frac;
        } else {
            return false;
        }
    }
    if (*s != '\0') {
        if (*(++s) == '\0') return false;
        for (; *s; ++s) {
            if (*s < '0' || *s > '9') return false;
            exp10 = exp10 * 10 + (*s - '0');
            if (exp10 > 60) return false;
        }
    }
    if (frac > 0) exp10 -= frac;
    if (exp10 < 0) return false;
    while (exp10-- > 0) {
        if (mant > ((UInt128)-1) / 10) return false;
        mant *= 10;
    }
    out = mant;
    return true;
}

// Decimal digits of a UInt128, for printing.
static std::string toString(UInt128 v) {
    if (v == 0) return "0";
    std::string s;
    for (; v > 0; v /= 10) s.insert(s.begin(), (char)('0' + (int)(v % 10)));
    return s;
}

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <n> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  n                    integer with 10^8 <= n <= 10^26;"
                 " scientific notation accepted (1e22, 2.5e21)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --profile, -p        print timing breakdown by computation phase" << std::endl;
    std::cerr << "  --segment-cap <len>  cap on stored Loop 2 sieve entries"
                 " (default: 12000000000, about 12 GB)" << std::endl;
    std::cerr << "  --loop01-int32-segment-size <len>"
                 "  fixed Int32 Loop 0/1 entries (0: adaptive sizing)"
              << std::endl;
    std::cerr << "  --s1-int32-chunk <rows>"
                 "  Int32 outer-Q6-family S1 dynamic chunk (default: 8)"
              << std::endl;
    std::cerr << "  --u <value>          set the sieve truncation point directly" << std::endl;
    std::cerr << "  --u-factor <value>   set the u scaling factor"
                 " (default depends on LOOP2_SIEVE_P)" << std::endl;
    std::cerr << "  --nu-ratio <value>   S1/S2 split ratio"
                 " (compiled default: " << MertensHurstDefaultNuRatio()
              << ")" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  --u and --u-factor are mutually exclusive." << std::endl;
    std::cerr << "  Bounds: 0 < u < n. u-factor > 0. nu-ratio > 0. Hard caps on u are enforced per build (see INPUT_BOUNDS.md)." << std::endl;
    std::cerr << "  floor(nu-ratio * sqrt(n)) must exceed 13860." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    bool profile = false;
    UInt64 segmentCap = 12000000000ULL;
    UInt64 loop01Int32SegmentSize = 0;
    UInt32 s1Int32Chunk = 8;
    UInt64 uOverride = 0;
    double uFactor = 0.0;
    double nuRatio = MertensHurstDefaultNuRatio();
    const char* nstr = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0 || std::strcmp(argv[i], "-p") == 0) {
            profile = true;
        } else if (std::strcmp(argv[i], "--segment-cap") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --segment-cap requires a value." << std::endl;
                return 1;
            }
            UInt128 capParsed;
            if (!parseNumber(argv[++i], capParsed) || capParsed == 0 || (capParsed >> 64) != 0) {
                std::cerr << "Error: --segment-cap must be a positive integer." << std::endl;
                return 1;
            }
            segmentCap = static_cast<UInt64>(capParsed);
        } else if (std::strcmp(argv[i], "--loop01-int32-segment-size") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --loop01-int32-segment-size requires a value."
                          << std::endl;
                return 1;
            }
            UInt128 sizeParsed;
            if (!parseNumber(argv[++i], sizeParsed) || (sizeParsed >> 64) != 0) {
                std::cerr << "Error: --loop01-int32-segment-size must be a nonnegative integer."
                          << std::endl;
                return 1;
            }
            loop01Int32SegmentSize = static_cast<UInt64>(sizeParsed);
        } else if (std::strcmp(argv[i], "--s1-int32-chunk") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --s1-int32-chunk requires a value."
                          << std::endl;
                return 1;
            }
            UInt128 chunkParsed;
            if (!parseNumber(argv[++i], chunkParsed) || chunkParsed == 0
                || chunkParsed > UInt128(std::numeric_limits<int>::max())) {
                std::cerr << "Error: --s1-int32-chunk must be an integer in [1, INT_MAX]."
                          << std::endl;
                return 1;
            }
            s1Int32Chunk = static_cast<UInt32>(chunkParsed);
        } else if (std::strcmp(argv[i], "--u") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --u requires a value." << std::endl;
                return 1;
            }
            UInt128 uParsed;
            if (!parseNumber(argv[++i], uParsed) || uParsed == 0 || (uParsed >> 64) != 0) {
                std::cerr << "Error: --u must be a positive integer." << std::endl;
                return 1;
            }
            uOverride = static_cast<UInt64>(uParsed);
        } else if (std::strcmp(argv[i], "--u-factor") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --u-factor requires a value." << std::endl;
                return 1;
            }
            uFactor = std::atof(argv[++i]);
            if (uFactor <= 0.0) {
                std::cerr << "Error: --u-factor must be positive (got "
                             << uFactor << ")." << std::endl;
                return 1;
            }
        } else if (std::strcmp(argv[i], "--nu-ratio") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --nu-ratio requires a value." << std::endl;
                return 1;
            }
            nuRatio = std::atof(argv[++i]);
            if (nuRatio <= 0.0) {
                std::cerr << "Error: --nu-ratio must be positive (got "
                             << nuRatio << ")." << std::endl;
                return 1;
            }
        } else {
            nstr = argv[i];
        }
    }

    if (!nstr) {
        std::cerr << "Error: no value for n provided." << std::endl;
        return 1;
    }

    // Validate mutual exclusivity
    if (uOverride > 0 && uFactor > 0.0) {
        std::cerr << "Error: --u and --u-factor are mutually exclusive. "
                  << "Use --u to set u directly, or --u-factor to set"
                     " the scaling factor." << std::endl;
        return 1;
    }

    UInt128 n;
    if (!parseNumber(nstr, n)) {
        std::cerr << "Error: could not parse n = '" << nstr << "'." << std::endl;
        return 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);
    Int64 result = MertensHurst(
        n, profile, segmentCap, uOverride, uFactor, nuRatio,
        loop01Int32SegmentSize, s1Int32Chunk
    );
    gettimeofday(&end, NULL);

    double t = (Int64)(end.tv_sec) - (Int64)(start.tv_sec)
             + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "M(" << toString(n) << ") = " << result << " in " << t << " seconds" << std::endl;
    return 0;
}
