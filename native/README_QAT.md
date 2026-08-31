# Optional Intel QAT/QATzip gzip extension

`qat_codec.c` implements the optional `XTA._qat_codec` companion used by
the preferred `YOLO_TTA_NRRD_MEMBER_CODEC=auto` path and explicit `qat`
selection. The base package imports it lazily, so a CPU-only installation does
not need QATzip or QAT hardware.

## Build requirements

- Linux x86-64 and CPython 3.10 or newer.
- QATzip public API 2.5 or newer (QATzip 1.3.2+), including `qatzip.h`,
  `libqatzip`, and the `qatzip` pkg-config file.
- A C11 compiler and pthreads.
- A configured QAT compression service, usable hardware instances, driver
  access, and any pinned-memory support required by the selected QAT stack.

The source enforces `QATZIP_API_VERSION >= 20500`. The build consumes:

```bash
pkg-config --cflags --libs qatzip  # normally supplies -lqatzip
```

Intel's supported distro packages are `qatzip qatzip-devel` on RHEL-family
systems and `qatzip libqatzip3 libqatzip-dev` on Debian-family systems.

Normal source builds use `XTA_BUILD_INTEL=auto`: QAT is included only
when `pkg-config` discovers QATzip. A missing development package therefore
still produces a CPU-capable wheel. To require QAT and fail the build if it
cannot be compiled:

```bash
XTA_BUILD_INTEL=qat python -m pip wheel . --no-deps
```

`XTA_BUILD_INTEL=none` forces a build without this extension.

## Runtime contract

The binding creates one QATzip session per calling executor thread and always
selects `QZ_DEFLATE_GZIP`, never Intel's extended gzip format. It requests
compression-only operation, sets `sw_backup=0`, disables software-only and
latency-sensitive selection, and uses QATzip's 128-byte minimum threshold.
Inputs below that threshold are rejected before a native request.

Each request uses QATzip's documented `qzMaxCompressedLength()` bound and
`qzCompressExt(..., last=1)`. The binding rejects a non-success status, partial
input consumption, a reported software or timeout bit, loss of the public
hardware-session state, or nonstandard/extended gzip framing. The Python
control plane then performs a hardware-only gzip round trip on every worker
thread before an output opens.

QATzip 1.3.2 has two important upstream observability gaps:

- `qzGetStatus()` returns success without populating `QzStatus_T`, and the
  software-component version queries are stubs in the upstream implementation.
  Admission therefore uses successful no-software `qzInit` and
  `qzSetupSessionDeflate` plus the public `QzSession_T.hw_session_stat == QZ_OK`.
  Status fields remain best-effort diagnostics, not an admission requirement.
- Its synchronous Deflate implementation does not reliably populate
  `qzCompressExt`'s execution-provenance bits. The binding still inspects and
  rejects those bits when present, but does not claim that they alone prove
  offload. Hardware-only admission relies on disabled fallback and sensitive
  mode, an at-or-above-threshold input, a proven hardware session, and native
  request success/full consumption.

QATzip can silently route level 9 to its software provider on older QAT stacks
without exposing trustworthy provenance. This binding consequently advertises
only levels 1 through 8 and rejects level 9. The normal gzip level 3 maps to QAT
level 1. Restoring level 9 requires an upstream API/version that reports either
hardware generation or dependable per-request execution provenance.

The public QATzip session API does not provide deterministic instance NUMA
binding, so the codec uses library-managed placement. Current QATzip status
APIs also do not expose a reliable instance count, hardware generation,
physical gzip-member count, or queue-busy count.
The binding therefore uses a conservative concurrency of one when the optional
device count is absent and reports unavailable metrics as `None`.

QATzip 1.3.2 retains process-global driver and lock state across `fork()` and
has no public child-reset API. The extension resets only its own counters/lock,
never calls inherited QATzip cleanup, and reports hardware unavailable in a
post-fork child. Use a `spawn`/`exec` worker for QAT. Explicit same-thread
cleanup and pthread TLS destructors tear sessions down in their owning process.

## Selection and opt-out

```bash
# Default: try QAT, then the existing CPU codec chain before output opens.
export YOLO_TTA_NRRD_MEMBER_CODEC=auto

# Opt out of all accelerator probing.
export YOLO_TTA_NRRD_MEMBER_CODEC=cpu

# Require QAT and fail closed if probing, per-thread preflight, or KAT fails.
export YOLO_TTA_NRRD_MEMBER_CODEC=qat
```

No backend switch is allowed after an NRRD begins streaming.

## Verification

The hardware-free contract suite compiles the production source against a fake
QATzip provider on Linux and exercises status stubs, no-hardware setup,
round-trip framing, hidden-software signals, partial consumption, timeouts,
TLS cleanup, and post-fork rejection:

```bash
python -m unittest tests.test_qat_native_binding -v
```

On the provisioned production host, force the extension build and run the
hardware-gated smoke test/microbenchmark:

```bash
XTA_BUILD_INTEL=qat python -m pip install .
python tools/intel_accelerator_selftest.py \
  --backend qat --qat-level 1 --size-mib 16 --iterations 5
```

The command probes the installed binding, preflights the calling thread,
performs hardware-only gzip round trips, checks that software fallback remains
zero, and prints capability, ratio, throughput, and native-counter data. Run it
inside the actual Slurm allocation so device permissions, QAT configuration,
NUMA placement, and pinned-memory policy match production.
