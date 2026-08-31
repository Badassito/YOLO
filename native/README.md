# Optional Intel native extensions

## QAT/QATzip gzip

`qat_codec.c` builds `XTA._qat_codec` on Linux x86-64 against QATzip
1.3.2 or newer (public API 2.5+, `qatzip.h`, pkg-config package `qatzip`, and
`libqatzip`) with C11 and pthreads. Automatic source builds include it when
`pkg-config` discovers QATzip; CPU-only builds remain valid. Force a build, and
fail if the dependency is absent, with:

```bash
XTA_BUILD_INTEL=qat python -m pip wheel . --no-deps
```

At runtime, `YOLO_TTA_NRRD_MEMBER_CODEC=auto` prefers QAT before the CPU codec
chain. `YOLO_TTA_NRRD_MEMBER_CODEC=cpu` is the opt-out that never probes QAT,
and `qat` explicitly requires it. The binding uses one session per executor
thread, `QZ_DEFLATE_GZIP`, `sw_backup=0`, compression-only direction, disabled
software-only/latency-sensitive selection, the 128-byte hardware threshold,
and full-consumption/status/framing checks. Every worker performs a hardware
gzip round trip before output opens.

Only QAT levels 1 through 8 are advertised. QATzip 1.3.2 can route level 9 to
software on older stacks without trustworthy provenance, so level 9 fails
closed. The same release leaves `qzGetStatus` and component-version reporting
partially stubbed; the binding uses the public hardware-session status after a
no-software setup and treats those other fields as optional diagnostics.

QATzip also retains process-global driver/lock state across `fork()` without a
public child reset. A post-fork child reports QAT unavailable and never cleans
up inherited QATzip state; use `spawn` or `exec` for accelerator workers.

Run the production-host smoke test inside the actual Slurm allocation:

```bash
python tools/intel_accelerator_selftest.py \
  --backend qat --qat-level 1 --size-mib 16 --iterations 5
```

See `README_QAT.md` for the precise proof model, upstream observability gaps,
package names, fake-provider tests, NUMA/concurrency limitations, and complete
deployment notes.

## IAA/QPL gzip

`qpl_codec.c` is the explicit, hardware-only `XTA._qpl_codec` backend.
See `README_QPL.md` for its QPL 1.9.0 build floor, runtime work-queue checks,
level-1 limitation, descriptor-safe chunking, fork policy, telemetry, and the
real-hardware smoke command. IAA remains outside the automatic codec chain and
is selected with `YOLO_TTA_NRRD_MEMBER_CODEC=iaa`.

## DSA workspace copy

`dsa_copy.c` is a project-owned CPython binding to the maintained Linux `idxd`
user-work-queue character-device ABI. It does not link DML, DTO, DPDK, or a
software memcpy fallback. Build it on Linux x86-64 with Python development tools
and a distro kernel-UAPI package that provides `linux/idxd.h`:

```bash
XTA_BUILD_INTEL=dsa python -m pip install .
```

An explicit build fails if the header or platform is unsuitable. The ordinary
wheel remains usable without the extension. At runtime, the host must load the
`idxd` driver and provision at least one enabled `type=user` DSA work queue with
`block_on_fault=1` (and, when present, `prs_disable=0`) plus an accessible
`/dev/dsa/wqN.M` character device. `accel-config` may be used to provision the
device, but it is not a linked runtime dependency.

Workspace copy remains CPU-only by default. Enable opportunistic offload with:

```bash
export YOLO_TTA_WORKSPACE_COPY_BACKEND=auto
export YOLO_TTA_DSA_MIN_MIB=64
export YOLO_TTA_DSA_MAX_INFLIGHT=32
# Optional; otherwise the binding selects the lowest-numbered queue local to
# the calling CPU's NUMA node.
export YOLO_TTA_DSA_WQ=/dev/dsa/wq0.0
```

`auto` uses DSA only for eligible contiguous, non-overlapping anonymous RAM,
memfd, or tmpfs copies. `dsa` is the fail-closed diagnostic mode. The binding
splits copies at the queue's sysfs `max_transfer_size`, submits bounded batches,
waits for every completion record, and keeps both Python buffers alive through
drain. On a submitted-request failure it returns only after all observed
records are terminal or after Linux `close(2)` has completed the `idxd` cdev
release path, which waits out the work-queue context. Only then may `auto`
perform its full CPU recovery copy.

A companion binding that reports `drained=false` is treated as a catastrophic
contract violation: the Python manager raises without starting a CPU copy and
quarantines strong references to both buffers. It also defers owned-file
close/unlink until a later manager shutdown obtains an explicit drain proof, so
an uncertain DMA target is never unmapped underneath the device.

The extension registers a `pthread_atfork` gate. Fork preparation waits until
all capability probes and copies have closed their cdev descriptors, so a child
cannot inherit an active PASID/work-queue context. The child then lazily replaces
the inherited Python lock and process counters without acquiring or freeing a
lock that might have belonged to a vanished parent thread.

The drain contract is implemented in
`drivers/dma/idxd/cdev.c::idxd_file_dev_release()`: shared queues drain the
caller's PASID, while dedicated queues drain/disable the queue before the SVA
binding and per-open context are released. Upstream commit
`e6fd6d7e5f0fe4a17a08e892afb5db800e7794ec` moved that existing close-time
drain into the per-open device release; it did not introduce the contract, so
the binding does not guess at a kernel-version cutoff. The hardware smoke test
below is the deployment admission check.

Run the hardware-gated smoke test on the provisioned production host:

```bash
YOLO_TTA_DSA_WQ=/dev/dsa/wq0.0 \
  python tools/intel_accelerator_selftest.py --backend dsa --size-mib 64 --iterations 5
```

The report includes the selected queue, NUMA identity, descriptor results, and
measured throughput. Compare `auto` against `cpu` on the real pipeline before
lowering the default threshold.
