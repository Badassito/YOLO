# Optional Intel IAA/QPL gzip extension

`qpl_codec.c` implements the optional `volume_tta._qpl_codec` companion used by
explicit `YOLO_TTA_NRRD_MEMBER_CODEC=iaa` selection. Importing the base package
does not import this extension, and a base wheel still builds without QPL.

## Supported build and runtime

- Linux x86-64.
- Intel QPL **1.9.0 or newer**, including `qpl/qpl.h`, `libqpl`, and the `qpl`
  pkg-config file. The source enforces the minimum through QPL's generated
  `QPL_VERSION_MAJOR` and `QPL_VERSION_MINOR` macros.
- A C11 compiler, Python development headers, `pkg-config`, pthreads, `libdl`,
  and the C++ runtime (`libstdc++` for GCC-family builds). QPL 1.9's `qpl.pc`
  advertises only `-lqpl`; it does not include these companion link flags.
- At runtime, `libaccel-config.so.1` (or `libaccel-config.so`) and at least one
  accessible, enabled, shared user IAA (`iax`) work queue with compression
  opcode `0x43` enabled and a nonzero maximum transfer size.

The extension does not require libaccel-config headers or a direct build-time
link. It loads the small inventory ABI with `dlopen()` so an installed extension
can import and report `hardware_available=False` safely on a host without IAA.

QPL 1.9.0 is the deliberate compatibility floor rather than a best-effort claim
for older releases. It is the first release whose generated public version
header provides the component macros used by this build guard, and it is the
API/behavior baseline audited for safe Deflate sizing, NUMA selection, terminal
page-fault statuses, and hardware-path level validation.

## Build

To force a QPL build and fail if its development package is missing:

```bash
pkg-config --atleast-version=1.9.0 qpl
VOLUME_TTA_BUILD_INTEL=qpl python -m pip wheel . --no-deps
```

The normal `auto` build includes the extension only when pkg-config discovers
QPL; `VOLUME_TTA_BUILD_INTEL=none` forces a pure-Python build.

## Runtime contract

The module uses `qpl_path_hardware` exclusively. It never initializes QPL's
`auto` or software paths. Every physical submission is a complete independent
member with `FIRST | LAST | GZIP_MODE`, dynamic Huffman generation, and QPL
verification omitted. Omitting QPL verification avoids a host decompression
pass; the Python adapter's per-worker known-answer test still validates RFC-1952
round-trip behavior before output opens.

Only compression level `1` is advertised. QPL 1.9 rejects `qpl_high_level`
(public level `3`) on `qpl_path_hardware`, so accepting it would make an explicit
IAA request fail after selection.

For each eligible NUMA set, the binding derives a conservative input chunk from
the work queues' maximum transfer size. Both the source and the destination
capacity must fit a descriptor. The destination capacity is calculated with
`qpl_get_safe_deflate_compression_buffer_size()` plus the documented 10-byte
gzip header and 8-byte trailer. Larger logical requests become ordered,
concatenated gzip members.

Each executor thread owns one initialized `qpl_job`. Native calls release the
GIL, thread-exit and explicit cleanup finalize the job, and PID guards discard
inherited jobs after `fork()` without finalizing parent state. A read/write
at-fork gate prevents the parent from forking in the middle of a native call
while allowing normal compression calls to run in parallel. QPL 1.9 exposes no
supported child reset for its process-global dispatcher, work-queue mappings,
or PASID-related state, so an inheriting fork child reports IAA unavailable and
fails closed. Use the multiprocessing `spawn` method or `exec` before selecting
IAA in a child process.

QPL Deflate has a 4-KiB history window. Its output is valid gzip but can be
larger than QAT/libdeflate/ISA-L output for structured volume data; representative
compression-ratio and throughput measurement remains a production release gate.

## Hardware smoke test

On a configured IAA host, build/install the extension and run:

```bash
VOLUME_TTA_TEST_IAA_HARDWARE=1 \
  python -m unittest tests.test_qpl_codec_contract.QplHardwareSmokeTests -v
```

The smoke test requires a real hardware capability probe, performs a 3-MiB-plus
hardware-only gzip round trip, checks the counters, and verifies that software
fallback remains zero. Without the opt-in variable the hardware tests skip;
the source-contract tests remain hardware-independent.

Useful initial diagnostics are:

```bash
pkg-config --modversion qpl
ldconfig -p | grep -E 'libqpl|libaccel-config'
accel-config list -i
YOLO_TTA_NRRD_MEMBER_CODEC=iaa python -c \
  'from volume_tta.intel_compression import probe_capabilities; print(probe_capabilities("iaa"))'
```

The production run should also confirm work-queue permissions in the Slurm job,
NUMA placement, terminal page-fault/queue-busy counters, and compressed-size
ratio against the CPU and QAT backends.
