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

# --- Final recovery (default: final value only) ---
# When 1, recover every square-free partial value by back substitution.
# The default obtains only M(n) by direct Mobius inversion. Override with:
# make FULL_RECOVERY=1
FULL_RECOVERY ?= 0

# --- Inclusion-exclusion transform (default: outer Q=6 for S1 and S2) ---
# `make q2` builds the original all-Q=2 implementation as a reference.
S1_OUTER_Q6 ?= 1
S2_OUTER_Q6 ?= 1
COHERENT_Q6 ?= 0
COHERENT_PERIOD36 ?= 0
UNORDERED_S2 ?= 0
VALIDATE_UNORDERED_S2 ?= 0
VALIDATE_UNORDERED_S2_WIDE ?= 0
Q6_ZERO_COMPLETION ?= 0
Q6_ZERO_COMPLETION_VALIDATE ?= 0
Q6_COMPACT_HOT_STATE ?= 0
Q30_COUPLED ?= 0
Q210_COUPLED ?= 0

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
            -DMERTENSHURST_FULL_RECOVERY=$(FULL_RECOVERY) \
            -DMERTENSHURST_S1_OUTER_Q6=$(S1_OUTER_Q6) \
            -DMERTENSHURST_S2_OUTER_Q6=$(S2_OUTER_Q6) \
            -DMERTENSHURST_COHERENT_Q6=$(COHERENT_Q6) \
            -DMERTENSHURST_COHERENT_PERIOD36=$(COHERENT_PERIOD36) \
            -DMERTENSHURST_UNORDERED_S2=$(UNORDERED_S2) \
            -DMERTENSHURST_VALIDATE_UNORDERED_S2=$(VALIDATE_UNORDERED_S2) \
            -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=$(VALIDATE_UNORDERED_S2_WIDE) \
            -DMERTENSHURST_Q6_ZERO_COMPLETION=$(Q6_ZERO_COMPLETION) \
            -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=$(Q6_ZERO_COMPLETION_VALIDATE) \
            -DMERTENSHURST_Q6_COMPACT_HOT_STATE=$(Q6_COMPACT_HOT_STATE) \
            -DMERTENSHURST_Q30_COUPLED=$(Q30_COUPLED) \
            -DMERTENSHURST_Q210_COUPLED=$(Q210_COUPLED)

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024 -DSIEVE_STRIDE_LOG=7"
CXXFLAGS += $(EXTRA_CXXFLAGS)

# Named profiles remove every formula/state control before appending one
# complete contract. Architecture and sieve tuning flags remain overridable.
CONTRACT_FILTERS = -DMERTENSHURST_FULL_RECOVERY=% \
                   -DMERTENSHURST_FULL_RECOVERY -UMERTENSHURST_FULL_RECOVERY \
                   -DMERTENSHURST_S1_OUTER_Q6=% \
                   -DMERTENSHURST_S1_OUTER_Q6 -UMERTENSHURST_S1_OUTER_Q6 \
                   -DMERTENSHURST_S2_OUTER_Q6=% \
                   -DMERTENSHURST_S2_OUTER_Q6 -UMERTENSHURST_S2_OUTER_Q6 \
                   -DMERTENSHURST_COHERENT_Q6=% \
                   -DMERTENSHURST_COHERENT_Q6 -UMERTENSHURST_COHERENT_Q6 \
                   -DMERTENSHURST_COHERENT_PERIOD36=% \
                   -DMERTENSHURST_COHERENT_PERIOD36 -UMERTENSHURST_COHERENT_PERIOD36 \
                   -DMERTENSHURST_UNORDERED_S2=% \
                   -DMERTENSHURST_UNORDERED_S2 -UMERTENSHURST_UNORDERED_S2 \
                   -DMERTENSHURST_VALIDATE_UNORDERED_S2=% \
                   -DMERTENSHURST_VALIDATE_UNORDERED_S2 -UMERTENSHURST_VALIDATE_UNORDERED_S2 \
                   -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=% \
                   -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE -UMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE \
                   -DMERTENSHURST_Q6_ZERO_COMPLETION=% \
                   -DMERTENSHURST_Q6_ZERO_COMPLETION -UMERTENSHURST_Q6_ZERO_COMPLETION \
                   -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=% \
                   -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE -UMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE \
                   -DMERTENSHURST_Q6_COMPACT_HOT_STATE=% \
                   -DMERTENSHURST_Q6_COMPACT_HOT_STATE -UMERTENSHURST_Q6_COMPACT_HOT_STATE \
                   -DMERTENSHURST_Q30_COUPLED=% \
                   -DMERTENSHURST_Q30_COUPLED -UMERTENSHURST_Q30_COUPLED \
                   -DMERTENSHURST_Q210_COUPLED=% \
                   -DMERTENSHURST_Q210_COUPLED -UMERTENSHURST_Q210_COUPLED
FIXED_BASE_CXXFLAGS = $(filter-out $(CONTRACT_FILTERS),$(CXXFLAGS))

Q2_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
              -DMERTENSHURST_FULL_RECOVERY=0 \
              -DMERTENSHURST_S1_OUTER_Q6=0 \
              -DMERTENSHURST_S2_OUTER_Q6=0 \
              -DMERTENSHURST_COHERENT_Q6=0 \
              -DMERTENSHURST_COHERENT_PERIOD36=0 \
              -DMERTENSHURST_UNORDERED_S2=0 \
              -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
              -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
              -DMERTENSHURST_Q6_ZERO_COMPLETION=0 \
              -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
              -DMERTENSHURST_Q6_COMPACT_HOT_STATE=0 \
              -DMERTENSHURST_Q30_COUPLED=0

FULL_RECOVERY_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                         -DMERTENSHURST_FULL_RECOVERY=1 \
                         -DMERTENSHURST_S1_OUTER_Q6=1 \
                         -DMERTENSHURST_S2_OUTER_Q6=1 \
                         -DMERTENSHURST_COHERENT_Q6=0 \
                         -DMERTENSHURST_COHERENT_PERIOD36=0 \
                         -DMERTENSHURST_UNORDERED_S2=0 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION=0 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
                         -DMERTENSHURST_Q6_COMPACT_HOT_STATE=0 \
                         -DMERTENSHURST_Q30_COUPLED=0

UNORDERED_S2_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                        -DMERTENSHURST_FULL_RECOVERY=0 \
                        -DMERTENSHURST_S1_OUTER_Q6=1 \
                        -DMERTENSHURST_S2_OUTER_Q6=1 \
                        -DMERTENSHURST_COHERENT_Q6=1 \
                        -DMERTENSHURST_COHERENT_PERIOD36=1 \
                        -DMERTENSHURST_UNORDERED_S2=1 \
                        -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                        -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                        -DMERTENSHURST_Q6_ZERO_COMPLETION=0 \
                        -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
                        -DMERTENSHURST_Q6_COMPACT_HOT_STATE=0 \
                        -DMERTENSHURST_Q30_COUPLED=0

ZERO_COMPLETE_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                         -DMERTENSHURST_FULL_RECOVERY=0 \
                         -DMERTENSHURST_S1_OUTER_Q6=1 \
                         -DMERTENSHURST_S2_OUTER_Q6=1 \
                         -DMERTENSHURST_COHERENT_Q6=1 \
                         -DMERTENSHURST_COHERENT_PERIOD36=1 \
                         -DMERTENSHURST_UNORDERED_S2=1 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
                         -DMERTENSHURST_Q6_COMPACT_HOT_STATE=0 \
                         -DMERTENSHURST_Q30_COUPLED=0

ZERO_COMPLETE_VALIDATE_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                                  -DMERTENSHURST_FULL_RECOVERY=0 \
                                  -DMERTENSHURST_S1_OUTER_Q6=1 \
                                  -DMERTENSHURST_S2_OUTER_Q6=1 \
                                  -DMERTENSHURST_COHERENT_Q6=1 \
                                  -DMERTENSHURST_COHERENT_PERIOD36=1 \
                                  -DMERTENSHURST_UNORDERED_S2=1 \
                                  -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                                  -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                                  -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                                  -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=1 \
                                  -DMERTENSHURST_Q6_COMPACT_HOT_STATE=0 \
                                  -DMERTENSHURST_Q30_COUPLED=0

OPTIMIZED_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                     -DMERTENSHURST_FULL_RECOVERY=0 \
                     -DMERTENSHURST_S1_OUTER_Q6=1 \
                     -DMERTENSHURST_S2_OUTER_Q6=1 \
                     -DMERTENSHURST_COHERENT_Q6=1 \
                     -DMERTENSHURST_COHERENT_PERIOD36=1 \
                     -DMERTENSHURST_UNORDERED_S2=1 \
                     -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                     -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                     -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                     -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
                     -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
                     -DMERTENSHURST_Q30_COUPLED=0

OPTIMIZED_VALIDATE_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                              -DMERTENSHURST_FULL_RECOVERY=0 \
                              -DMERTENSHURST_S1_OUTER_Q6=1 \
                              -DMERTENSHURST_S2_OUTER_Q6=1 \
                              -DMERTENSHURST_COHERENT_Q6=1 \
                              -DMERTENSHURST_COHERENT_PERIOD36=1 \
                              -DMERTENSHURST_UNORDERED_S2=1 \
                              -DMERTENSHURST_VALIDATE_UNORDERED_S2=1 \
                              -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=1 \
                              -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                              -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=1 \
                              -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
                              -DMERTENSHURST_Q30_COUPLED=0

Q30_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
               -DMERTENSHURST_FULL_RECOVERY=0 \
               -DMERTENSHURST_S1_OUTER_Q6=1 \
               -DMERTENSHURST_S2_OUTER_Q6=1 \
               -DMERTENSHURST_COHERENT_Q6=1 \
               -DMERTENSHURST_COHERENT_PERIOD36=1 \
               -DMERTENSHURST_UNORDERED_S2=1 \
               -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
               -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
               -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
               -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
               -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
               -DMERTENSHURST_Q30_COUPLED=1

Q30_VALIDATE_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                        -DMERTENSHURST_FULL_RECOVERY=0 \
                        -DMERTENSHURST_S1_OUTER_Q6=1 \
                        -DMERTENSHURST_S2_OUTER_Q6=1 \
                        -DMERTENSHURST_COHERENT_Q6=1 \
                        -DMERTENSHURST_COHERENT_PERIOD36=1 \
                        -DMERTENSHURST_UNORDERED_S2=1 \
                        -DMERTENSHURST_VALIDATE_UNORDERED_S2=1 \
                        -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=1 \
                        -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                        -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=1 \
                        -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
                        -DMERTENSHURST_Q30_COUPLED=1

Q210_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                -DMERTENSHURST_FULL_RECOVERY=0 \
                -DMERTENSHURST_S1_OUTER_Q6=1 \
                -DMERTENSHURST_S2_OUTER_Q6=1 \
                -DMERTENSHURST_COHERENT_Q6=1 \
                -DMERTENSHURST_COHERENT_PERIOD36=1 \
                -DMERTENSHURST_UNORDERED_S2=1 \
                -DMERTENSHURST_VALIDATE_UNORDERED_S2=0 \
                -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=0 \
                -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=0 \
                -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
                -DMERTENSHURST_Q30_COUPLED=1 \
                -DMERTENSHURST_Q210_COUPLED=1

Q210_VALIDATE_CXXFLAGS = $(FIXED_BASE_CXXFLAGS) \
                         -DMERTENSHURST_FULL_RECOVERY=0 \
                         -DMERTENSHURST_S1_OUTER_Q6=1 \
                         -DMERTENSHURST_S2_OUTER_Q6=1 \
                         -DMERTENSHURST_COHERENT_Q6=1 \
                         -DMERTENSHURST_COHERENT_PERIOD36=1 \
                         -DMERTENSHURST_UNORDERED_S2=1 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2=1 \
                         -DMERTENSHURST_VALIDATE_UNORDERED_S2_WIDE=1 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION=1 \
                         -DMERTENSHURST_Q6_ZERO_COMPLETION_VALIDATE=1 \
                         -DMERTENSHURST_Q6_COMPACT_HOT_STATE=1 \
                         -DMERTENSHURST_Q30_COUPLED=1 \
                         -DMERTENSHURST_Q210_COUPLED=1
Q210_SANITIZE_CXXFLAGS = $(filter-out -O3 -flto -DNDEBUG,$(Q210_VALIDATE_CXXFLAGS)) \
                         -O1 -g -fsanitize=address,undefined \
                         -fno-omit-frame-pointer
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

.PHONY: all q2 s2-unordered q6-zero-complete q6-zero-complete-validate \
        optimized optimized-validate q30-coupled q30-coupled-validate \
        q210-coupled q210-coupled-validate q210-coupled-sanitize \
        full-recovery clean

all: $(BUILDDIR)/mertens

q2: $(BUILDDIR)/mertens_q2

s2-unordered: $(BUILDDIR)/mertens_s2_unordered

q6-zero-complete: $(BUILDDIR)/mertens_q6_zero_complete

q6-zero-complete-validate: $(BUILDDIR)/mertens_q6_zero_complete_validate

optimized: $(BUILDDIR)/mertens_optimized

optimized-validate: $(BUILDDIR)/mertens_optimized_validate

q30-coupled: $(BUILDDIR)/mertens_q30_coupled

q30-coupled-validate: $(BUILDDIR)/mertens_q30_coupled_validate

q210-coupled: $(BUILDDIR)/mertens_q210_coupled

q210-coupled-validate: $(BUILDDIR)/mertens_q210_coupled_validate

q210-coupled-sanitize: $(BUILDDIR)/mertens_q210_coupled_sanitize

full-recovery: $(BUILDDIR)/mertens_full_recovery

$(BUILDDIR)/mertens: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q2: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q2_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_s2_unordered: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(UNORDERED_S2_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q6_zero_complete: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(ZERO_COMPLETE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q6_zero_complete_validate: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(ZERO_COMPLETE_VALIDATE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_optimized: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(OPTIMIZED_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_optimized_validate: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(OPTIMIZED_VALIDATE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q30_coupled: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q30_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q30_coupled_validate: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q30_VALIDATE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q210_coupled: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q210_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q210_coupled_validate: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q210_VALIDATE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(BUILDDIR)/mertens_q210_coupled_sanitize: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(Q210_SANITIZE_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS) -fsanitize=address,undefined

$(BUILDDIR)/mertens_full_recovery: Makefile $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(FULL_RECOVERY_CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)
