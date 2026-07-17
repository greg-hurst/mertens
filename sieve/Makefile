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

# --- Sub-buckets: cache-banded hit application in the bucket scheduler ---
# Wide entries use SUB_SHIFT; narrow entries use NARROW_SUB_SHIFT.
# Override with: make SUB_BUCKETS=0
SUB_BUCKETS ?= 1
SUB_SHIFT   ?= 17
NARROW_SUB_SHIFT ?= 16

# --- Bucket entry layout: 1 = narrow/prime-only/divide-per-hit (DEFAULT, 4 B;
# current hits transiently cache-banded), 0 = wide/packed/divide-free (8 B).
# Narrow wins at scale on many-core ARM; on x86 the per-hit divide is costlier,
# so re-measure there. See PERFORMANCE.md section 6.
NARROW_ENTRY ?= 1

# --- Base flags (all platforms) ---
CXXFLAGS := -O3 -std=c++17 \
            -flto -fno-exceptions -fno-rtti \
            -frounding-math -finline-functions \
            -march=native -mtune=native \
            -fno-strict-overflow -DNDEBUG \
            -DUSE_DIVISION_FREE=$(DIVISION_FREE) \
            -DUSE_BUCKET_SIEVE=$(BUCKET_SIEVE) \
            -DSIEVE_SUB_BUCKETS=$(SUB_BUCKETS) \
            -DSIEVE_SUB_SHIFT=$(SUB_SHIFT) \
            -DSIEVE_NARROW_SUB_SHIFT=$(NARROW_SUB_SHIFT) \
            -DSIEVE_NARROW_ENTRY=$(NARROW_ENTRY)

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024 -DSIEVE_STRIDE_LOG=7"
CXXFLAGS += $(EXTRA_CXXFLAGS)
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

DEMODIR  = demo
BUILDDIR = build
SRC      = SegmentedMobiusSieve.cpp

DEMOS = $(BUILDDIR)/demo_mobius_core \
        $(BUILDDIR)/demo_mobius_segmented \
        $(BUILDDIR)/demo_mobius_values \
        $(BUILDDIR)/demo_mertens_core \
        $(BUILDDIR)/demo_mertens_segmented \
        $(BUILDDIR)/demo_mertens_value \
        $(BUILDDIR)/demo_mertens_values

.PHONY: all clean

all: $(DEMOS)

$(BUILDDIR)/%: $(DEMODIR)/%.cpp $(SRC) $(wildcard *.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I. $< $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)
