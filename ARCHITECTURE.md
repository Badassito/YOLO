# Volume TTA architecture

The `GPT-5.6-Sol-Ultra_v17.1.5_SLURM.py` filename remains a versioned compatibility
launcher. The implementation lives in the importable `volume_tta` package so spawned
processes resolve worker functions and data types through canonical module paths.

## Runtime modules

| Module | Ownership |
|---|---|
| `config` | CLI grammar, validated selections, channel/view request formats, version constants |
| `workspace` | scratch policy and environment-derived workspace settings |
| `runtime` | telemetry, NUMA primitives, memfd/workspace ownership, executors and process setup |
| `media` | ffprobe/ffmpeg, decode/resize readiness and source-volume lifecycle |
| `geometry` | affine transforms, view definitions, slicing and render-source geometry |
| `inference` | shared Ultralytics execution, mask payloads and inference cleanup |
| `cuda_backend` | CUDA-resident rendering and CUDA worker-side helpers |
| `cuda_interpolation` | Lazy CUDA bridge morphology/radius evaluation and crop-bounded painting |
| `workers` | module-level OpenVINO and CUDA worker entry points |
| `topology` | slice labeling, union-find and component metadata |
| `backprojection` | radial/tilted projection plans and source-space accumulation |
| `finalization` | source-volume fusion, object filtering and centerline processing |
| `interpolation` | interpolation planning, execution and sparse continuation |
| `cuda_d1` | D1 owner-GPU backprojection and packed source-space storage |
| `assembly` | completed-view preparation, tile gates and smoothing handoff |
| `outputs` | NRRD, TIFF/MKV, summaries and low-quality derivatives |
| `intel_compression` | dependency-light policy/adapter for optional QATzip and QPL companion extensions |
| `intel_dsa` | lazy policy, eligibility, drain transaction and lifecycle for optional Linux idxd workspace copy |
| `pipeline` | the existing production orchestration routine |

`volume_tta.__init__` is deliberately inert. In particular, it does not import OpenCV,
SciPy, Ultralytics, CUDA, OpenVINO or future accelerator runtimes. The compatibility
launcher and `python -m volume_tta` both use `volume_tta.__main__.run()`.

The package enforces an acyclic eager import graph. Lower-level subsystems are imported
explicitly; the small number of callbacks into a higher-level subsystem use a function-local
import. This keeps every module independently importable, including concurrent first
imports, without a global symbol registry or wildcard-import facade.

## Backend boundary

`volume_tta.inference_backends` contains dependency-free control-plane contracts. The
current CUDA/OpenVINO scheduler owns tuned leasing and hybrid policy. Future scheduler work
should adapt those workers to `InferenceBackend` rather than adding another backend-specific
branch to `pipeline.main()`.

The scheduler-facing unit is an `ExecutionTarget`, not a device or process:

- a current CUDA target represents one independently scheduled local GPU;
- a current OpenVINO target represents one independently scheduled socket-local process;
- a future collective accelerator target may represent several hosts and ranks while still
  emitting one lease completion to the global scheduler.

This permits future TPU scale-out without implying or implementing GPU scale-out. A CUDA
adapter should reject `host_count != 1`; collective rank scheduling and failure aggregation
belong inside the future collective backend adapter. The contract does not require an x86
controller: a collective adapter and its coordinator may run entirely within an Arm-hosted
accelerator allocation, and no OpenVINO/CPU inference backend needs to be registered there.

`DispatchLease` separates `logical_slice_count` from `execution_slice_count`, allowing a
future compiled backend to pad work to fixed buckets. `ArtifactRef` carries a URI rather
than array contents, so a later multi-host transport can use remotely resolvable artifacts
without weakening the current local memfd/path implementation.

No TPU backend, TPU dependency, TPU CLI option or remote transport is registered today.
Unknown backends must fail closed through `BackendRegistry` rather than falling through to
CUDA behavior.

## Multiprocessing rules

- Launch through the compatibility script or `python -m volume_tta`; do not load modules
  under ad-hoc aliases.
- Worker targets remain module-level importable functions in `volume_tta.workers`.
- Keep Torch, CuPy, TensorRT, OpenVINO, Ultralytics and future TPU imports inside their
  owning runtime paths.
- Do not pass closures from `pipeline.main()` to `spawn` children.
- Preserve explicit per-worker initialization; spawn children do not inherit resolved
  process globals.
- Large data belongs behind artifact references or the existing descriptor/path transport,
  never directly in queue or future RPC messages.

## Verification

The test suite covers dependency-light CLI/config imports, configuration parsing,
cycle-safe package imports, accelerator policy/lifecycle, and collective-ready backend
contracts. The checked-in package statement inventory verifies unchanged definitions,
reviewed implementation changes, retired bindings, and explicitly reviewed local-import
seams.

Run the dependency-light checks with:

```powershell
python -m unittest discover -s tests -v
python tools/smoke_import.py
python tools/verify_package_inventory.py
```

Hardware-backed CUDA, TensorRT, OpenVINO, QAT, IAA, DSA and full data/model parity tests
still require the production environment and representative artifacts. Intel accelerator
build, provisioning, and admission instructions live in ``native/README.md``; run
``python tools/intel_accelerator_selftest.py --backend all`` on the target host.

Direct Radial/Tilted rendering and resident mask quantization use 32-by-8 pixel launches,
avoiding flattened-index division in the output kernels. The D1 path also derives each
slice's nonempty flag and exclusive bbox during final quantization. One tiny four-int record
per slice is copied with the task union, so D1 does not rescan the full device volume for
row/column extents. Batched bbox backprojection groups similarly sized slices into a bounded
number of launches, and Volta-or-newer kernels aggregate output-bit updates by warp/word
before the global atomic.

CUDA bridge painting is attempted by default only inside a leased, already-warm CUDA
worker and requires the CUDA extra's CuPy 13+ primitives. The first nonempty bounded
plan batch is rendered first through the exact parallel CPU painter and then replayed on
CUDA; CUDA keeps the remaining batches only when it is at least 5% faster. The replay is
safe because bridge painting is OR-idempotent. `YOLO_TTA_GPU_INTERPOLATION_RENDER_AUTOTUNE=0` forces
CUDA painting after admission, while `YOLO_TTA_GPU_INTERPOLATION=0` disables it entirely.
`YOLO_TTA_GPU_INTERPOLATION_REQUIRED=1` also forces CUDA and makes admission or execution
failure fatal instead of replaying work on CPU.

One interpolation GPU lease owns a bounded pool of non-default CuPy streams (four by
default, configurable with `YOLO_TTA_GPU_INTERPOLATION_STREAMS`). Disjoint destination
slices are dispatched in parallel up to that bound. The renderer lock covers shared cache
metadata and enqueue ordering only: metrics and destination D2H copies use explicitly pinned
host buffers with `blocking=False`, then one stream event is awaited without holding the
lock. The host crop is committed only after that event succeeds, preserving the failed-batch
CPU replay transaction. Cache entries
carry producer events for cross-stream dependencies, and every lease retains the device
objects it touched until its stream is quiescent so concurrent LRU eviction cannot recycle
in-flight storage.

The min-radius acceptance scan uses the existing no-GIL parallel CPU evaluator by default.
Production evidence from v17.1.1 showed that issuing each plan's CuPyX labeling, hole fill,
and EDT through one renderer lock serialized 128 planner threads and reduced throughput by
roughly fourfold. `YOLO_TTA_GPU_INTERPOLATION_RADIUS=1` keeps that CUDA radius path as an
explicit experiment, but opted-in planners now borrow independent streams and hold the
renderer lock only for shared cache/telemetry mutations. Radius failure is isolated from
painting: unless CUDA is required, the affected plan and remaining radius work return to CPU
while an otherwise healthy renderer may continue painting. Rendering coalesces per-section
device reductions into one scalar transfer per destination group instead of synchronizing
twice for every section.

Dedicated interpolation children and the main process do not create or claim CUDA contexts
unless `YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT=1` and, for the latter,
`YOLO_TTA_GPU_INTERPOLATION_MAIN_PROCESS=1` are both explicitly set. At admission, the
renderer requires `YOLO_TTA_GPU_INTERPOLATION_RESERVE_MIB` (1024 by default) of free VRAM
and withholds that amount when sizing its live SDF/section cache.
`YOLO_TTA_GPU_INTERPOLATION_CACHE_MIB` (1024 by default) caps retained device payloads;
temporary CuPy/CuPyX workspaces and allocator-pool blocks are outside that logical cache
limit and are released when the lease closes. The default global interpolation-pass limit
remains one because a production pass required about 117 GiB of host workspace. Per-pass
logs and runtime stats separately report radius/render backends, autotune timings, lock wait,
execution time, transfer categories, crop/patch pixels, cache eviction, fallback, and the
worker-visible physical CUDA token.

## Streaming inference completion and terminal fusion

OpenVINO request callbacks publish indexed completions into a bounded queue. A bounded
consumer pool, sized to the useful infer-request count, performs output decoding and
destination writes concurrently. Destination slices and aggregate statistics have separate
locks, callback/request draining is unconditional on failure, and competing failures are
reported in submission order so concurrency does not make error selection nondeterministic.

The scheduler treats each runtime TTA view as terminal when its full-frame/tile continuation
has retired. As soon as every variant of one physical view is terminal, ownership of that
group is detached from the inference registries, its variants are OR-collapsed, and any
Radial/Tilted projection runs while other views may still be inferencing. Completed physical
views feed one path-backed, single-writer source-space union reducer. Equal-geometry sparse
component layers are ORed before one restore per output slice, retaining the grouped G5
optimization. A one-credit dense handoff prevents finalizers from retaining multiple
source-sized volumes while waiting for the reducer.

Finalization and reducer futures are first-class scheduler dependencies, including terminal
coverage assertions at quiescence. The global centerline/smoothing stages still wait for the
complete union because their semantics span every view, but they no longer wait for a
separate post-inference collapse/backprojection/fusion phase.
