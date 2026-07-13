CXX      ?= c++

UNAME_S  := $(shell uname -s)

# --- Division-free mode: quotient cache + predictor (x86) vs direct division (ARM) ---
# Override with: make DIVISION_FREE=1  or  make DIVISION_FREE=0
UNAME_M  := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    DIVISION_FREE ?= 1
else
    DIVISION_FREE ?= 0
endif

# --- Bucket sieve: large-prime bucket scheduler (default: on) ---
# Override with: make BUCKET_SIEVE=0
BUCKET_SIEVE ?= 1

# --- Bucket entry layout: 1 = narrow/prime-only/divide-per-hit (DEFAULT, 4 B),
# 0 = wide/packed/divide-free (8 B). Narrow wins at scale on many-core ARM;
# wide may win where the per-hit divide is costlier. See
# ../sieve/PERFORMANCE.md section 6. Override with: make SIEVE_BUCKET_NARROW_ENTRY=0
SIEVE_BUCKET_NARROW_ENTRY ?= 1

# --- Fused finalize + Mertens prefix (default: on) ---
# When 1, Loop 2's in-place sieve skips the in-sieve mu finalization and folds
# it into the prefix scan (rides free on the scan's DRAM-bound load instead of
# a separate in-cache pass). Bit-identical M. Override with: make FUSED_FINALIZE=0
FUSED_FINALIZE ?= 1

# --- Inclusion-exclusion transform (default: outer Q=6 for S1 and S2) ---
# `make q2` builds the original all-Q=2 implementation as a reference.
S1_OUTER_Q6 ?= 1
S2_OUTER_Q6 ?= 1

# --- Base flags (all platforms) ---
CXXFLAGS := -O3 -std=c++17 \
            -I../sieve \
            -flto -fno-exceptions -fno-rtti \
            -frounding-math -finline-functions \
            -march=native -mtune=native \
            -fno-strict-overflow -DNDEBUG \
            -DUSE_DIVISION_FREE=$(DIVISION_FREE) \
            -DUSE_BUCKET_SIEVE=$(BUCKET_SIEVE) \
            -DSIEVE_NARROW_ENTRY=$(SIEVE_BUCKET_NARROW_ENTRY) \
            -DSIEVE_FUSED_FINALIZE=$(FUSED_FINALIZE) \
            -DMERTENSHURST_S1_OUTER_Q6=$(S1_OUTER_Q6) \
            -DMERTENSHURST_S2_OUTER_Q6=$(S2_OUTER_Q6)

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024 -DSIEVE_STRIDE_LOG=7"
CXXFLAGS += $(EXTRA_CXXFLAGS)
Q2_CXXFLAGS = $(filter-out -DMERTENSHURST_S1_OUTER_Q6=% \
                            -DMERTENSHURST_S2_OUTER_Q6=%,$(CXXFLAGS)) \
              -DMERTENSHURST_S1_OUTER_Q6=0 \
              -DMERTENSHURST_S2_OUTER_Q6=0
LDFLAGS  :=

# --- Platform-specific: OpenMP ---
ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    CXXFLAGS += -I$(BREW_PREFIX)/opt/libomp/include -Xpreprocessor -fopenmp
    LDFLAGS  += -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
endif

SRCDIR   = src
BUILDDIR = build
SIEVEDIR = ../sieve
SRCS     = $(SRCDIR)/MertensHurst.cpp $(SIEVEDIR)/SegmentedMobiusSieve.cpp $(SRCDIR)/main.cpp

.PHONY: all q2 clean

all: $(BUILDDIR)/mertens

q2: $(BUILDDIR)/mertens_q2

$(BUILDDIR)/mertens: $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q2: $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q2_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)
