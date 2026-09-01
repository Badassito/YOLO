# XTA architecture

`GPT-5.6-Sol-Ultra_v18.0.3_SLURM.py` is the sole versioned launcher. It, the
installed `xta` console script, and `python -m XTA` all dispatch
through `XTA.cli.run()`.
The implementation lives in the importable `XTA` package so spawned processes
resolve worker functions and data types through canonical module paths.

## Runtime modules

| Module | Ownership |
|---|---|
| `config` | CLI grammar, validated selections, channel/view request formats, version constants |
| `workspace` | scratch policy and environment-derived workspace settings |
| `runtime` | telemetry, NUMA primitives, memfd/workspace ownership, executors and process setup |
| `media` | ffprobe/ffmpeg, decode/resize readiness and source-volume lifecycle |
| `render_batch` | runtime frame-carrying `RenderBatch` values and exact model/image fan-out contracts; distinct from logical `RenderRequestBatch` planning |
| `geometry` | authoritative CPU forward renderer, affine/view/channel/seam/tile primitives, `RasterPlan` builders, slicing and render-source geometry |
| `gaussian` | one binary Gaussian numerical primitive shared by PTA preprocessing and TTA postprocessing |
| `inference` | shared Ultralytics execution, mask payloads and inference cleanup |
| `cuda_backend` | CUDA-resident rendering and CUDA worker-side helpers |
| `cuda_interpolation` | Lazy CUDA bridge morphology/radius evaluation and crop-bounded painting |
| `workers` | module-level OpenVINO and CUDA worker entry points |
| `topology` | slice labeling, union-find and component metadata |
| `backprojection` | radial/tilted projection plans and source-space accumulation |
| `finalization` | source-volume fusion, object filtering and centerline processing |
| `interpolation` | interpolation planning, execution and sparse continuation |
| `cuda_d1` | D1 owner-GPU backprojection and packed source-space storage |
| `cuda_finalization` | dependency-light distributed-binary contracts plus the opt-in v18.0.3 transactional multi-GPU keep tail |
| `assembly` | completed-view preparation, tile gates and smoothing handoff |
| `outputs` | NRRD, TIFF/MKV, summaries and low-quality derivatives |
| `unification.contracts` | dependency-light `ForwardSamplingPolicy`, digest-addressed `RasterPlan`, logical `RenderItem`/`RenderRequestBatch`, channel/tile and data-role contracts |
| `unification.sampling` | the executable forward-policy registry, fail-closed backend/role binding, execution records and canonical plan factory |
| `unification.channels`, `unification.tiles` | shared single-channel-layout expansion and strict grouped tile parsing |
| `unification.runtime`, `unification.views` | the shared TTA-authoritative physical-view compiler and dependency-light grouped view requests |
| `unification.context`, `unification.manifest`, `unification.tta_manifest` | launch context, atomic JSON publication, artifact identities and mode-qualified v18 manifests |
| `pta_config` | strict PTA-only v18 grammar, grouped geometry and mode-specific defaults |
| `pta_mode`, `pta_runtime` | dependency-light PTA validation followed by construction of the complete native runtime option contract |
| `pta_scheduler` | independently tested CUDA-owner layout, compatible-work packing, VRAM admission and deterministic OOM splitting |
| `pta_augmentation` | external PTA policy inspection, loading, deterministic thread-local construction and paired image/mask execution |
| `pta_dataset` | PTA candidate identity, deterministic augmentation-version planning, background policy and dataset splitting |
| `pta_rendering` | canonical spawn-pickled PTA render-plan values, CPU geometry caches and frame/tile rendering primitives |
| `pta_publication` | image/label encoding, atomic publication, canonical dataset image sink and candidate output paths |
| `pta_workers` | sole PTA shared-memory, worker-global, CPU/GPU task-entry, result and persistent-pool process owner |
| `pta` | PTA source discovery/preprocessing, geometry planning, dataset orchestration, reporting and manifest publication |
| `tta_mode` | production TTA runner entered only after mode-specific CLI validation |
| `tta_lifecycle` | outer TTA resource ownership, selected-run scratch cleanup and complete-manifest publication transaction |
| `tta_outputs` | single-use identity-preserving ownership and ordered teardown of settled TTA output artifacts |
| `tta_prediction` | lazy physical-view frame caching and bounded prediction-source build, staging and warmup queues |
| `tta_scheduler` | stateful TTA process-inference admission, hybrid/D1 ownership, CPU/GPU dispatch, result transport and accounting |
| `tta_terminal` | completed physical-view TTA collapse, one-time terminal backprojection and dense-union handoff credit |
| `cli` | dependency-light strict `--mode tta|pta` dispatcher |
| `intel_compression` | dependency-light policy/adapter for optional QATzip and QPL companion extensions |
| `intel_dsa` | lazy policy, eligibility, drain transaction and lifecycle for optional Linux idxd workspace copy |
| `pipeline` | production TTA preparation, completed-view assembly and output facade around the process scheduler |

### Incremental orchestration decomposition

v18.0.3 establishes explicit owners for TTA run resources/publication, prediction-source
preparation, terminal physical-view fusion, and PTA external augmentation. It also adds the
leaf PTA dataset-policy owner and a stateful `TtaScheduler` for process-inference admission,
hybrid CPU/GPU and D1 ownership, dynamic lease splitting, dispatch, result transport,
accounting, liveness checks, and worker shutdown. It also separates PTA's canonical render
graph, publication primitives, and complete worker/pool process state, plus establishes a
single-use TTA settled-artifact teardown owner. Compatibility names remain exact facade
aliases, while spawned PTA targets resolve directly through `XTA.pta_workers`.

The remaining TTA closure cluster now owns view preparation, P/B tile gates, consolidation,
physical-view reduction, output scheduling, and manifest construction. Those paths can expand
the existing `tta_outputs` boundary only after they produce settled assembly artifacts instead
of cross-reading live registries. PTA's worker extraction retains its prior CUDA/start-method
semantics verbatim and still requires physical-GPU qualification before release.

### v18.0.3 intra-node HGX experiments

Two experimental paths are dark by default and retain the established reference behavior:

- `YOLO_TTA_V1803_D1_OWNER_GROUPS=1` allows a D1 parent whose complete seed leases
  need no view-shadow writer to bind deterministic slice coverage to several idle CUDA
  workers. Admission is a centralized, atomic pre-dispatch reservation rather than a side
  effect of pending-task feasibility scans. v18.0.3 keeps one group active at a time, chooses
  the heaviest still-unclaimed eligible parent, and plans the next group after the prior
  reduction releases its participants; nonparticipants retain ordinary view-level parallelism.
  Each participant retains a dedicated IPC-exportable partial bitset until the scheduler
  acknowledges either CUDA-IPC/NVLink reduction or bounded host-path recovery, then confirms
  an explicit release acknowledgement from every participant before reusing the workers.
  `YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE` caps the participant count at the job-visible
  device count. The one-owner path remains the admission and execution fallback.
- `YOLO_TTA_V1803_GPU_RESIDENT_TAIL=1` tries the first Track-A transaction after the
  inference workers have drained: the settled host final union is uploaded into contiguous
  job-visible Z shards, exact 26-connected labels stay device-resident in memory-bounded
  3-D CCL blocks, and compact equivalence pairs are needed only at block/shard boundaries.
  Area and boundary metadata is resolved through the CPU union-find reference, and a separate filtered candidate is
  committed only after every GPU succeeds. `YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED=1`
  makes failure fatal for qualification; otherwise the untouched host union enters the
  established CPU `keep_objects` path.

`cuda_finalization.DistributedBinaryArtifact` is the common future handoff boundary for
host uint8 volumes, D1 bitsets, resident final-union shards, and a later multi-GPU
interpolation producer. The v18.0.3 Track-A implementation exercises the host-upload
adapter and resident keep transaction; direct resident final-union ingestion and Track B
interpolation remain incremental follow-on work. Every decomposition uses only devices
selected by the job and is written generically for one through eight GPUs.

Four-GPU HGX qualification established byte-exact output for both experiments but no
reproducible end-to-end wall-time benefit on the 30-view production workload. Track A reduced
its own `keep_objects` stage substantially, but cold parent CUDA startup and concurrent output
variance absorbed the gain. Size-2 D1 groups exercised 29.1 GiB of CUDA-IPC peer reads across
14 exact group transactions with zero host fallbacks, but the workload already had enough
independent views to occupy all devices. Both paths therefore remain opt-in infrastructure.
A future D1 revisit should promote only a genuine idle tail, and a future resident interpolation
producer may feed the distributed Track-A boundary without the cold host-upload adapter.

`XTA.__init__` is deliberately inert. In particular, it does not import OpenCV,
SciPy, Ultralytics, CUDA, OpenVINO or future accelerator runtimes. Every supported command
surface enters the same dependency-light mode dispatcher; TTA production dependencies are
loaded only after validation through `XTA.tta_mode.run()`.

The package enforces an acyclic eager import graph. Lower-level subsystems are imported
explicitly; the small number of callbacks into a higher-level subsystem use a function-local
import. This keeps every module independently importable, including concurrent first
imports, without a global symbol registry or wildcard-import facade.

## Unified forward-render contracts

`ForwardSamplingPolicy` is wired into production execution rather than serving as passive
documentation. The singleton policy declares coordinate/stage order, role-specific kernels and
boundaries, and the registered CPU/CUDA implementations. Runtime code resolves a backend and data
role through `require_forward_sampling()` and fails when the binding is absent instead of silently
substituting another kernel. `forward_sampling_execution_record()` serializes the same resolved
bindings into both mode manifests.

Every built-in full-frame or tile job receives an immutable `RasterPlan`. Its canonical record
contains mode, physical-view identity, in-plane variant, channel variant, output shape, optional
tile layout, frozen metadata, and the complete sampling policy. SHA-256 digests address both the
policy and plan. TTA builds plans beside its actual runtime sources; PTA builds plans beside its
publication plans and checks the embedded policy digest before rendering. Because mode and
mode-owned metadata are part of the canonical record, a PTA plan and analogous TTA plan are not
expected to share a digest. They are expected to resolve to the same implementation when backend,
data role, and built-in geometry are equivalent.

Two similarly named batch types sit on opposite sides of rendering:

- `unification.contracts.RenderRequestBatch` is a dependency-light tuple of logical
  `RenderItem`s. It contains frame addresses and plan identities, not arrays; empty PTA batches
  are valid.
- `render_batch.RenderBatch` carries the actual frames. A `RenderBatchItem.frame` must be the same
  object as the corresponding model-bound list element, and an attached logical request must
  match the batch plan digest. Synthetic Cartesian tail repeats and radial seam-extension slots
  remain explicitly marked so artifact sinks can omit them.

Only layout, dtype, normalization, and other backend-only conversion may occur after the
frame-carrying boundary. Geometry may not be independently reconstructed for an image sink.
CPU-backed main-process and OpenVINO/CUDA-worker slab sources implement this fan-out today;
device-resident CUDA/direct TensorRT-ring capture remains unfinished and unqualified.

Unification also uses shared operation primitives rather than duplicating mode adapters.
`unification.channels` expands the canonical TTA channel grammar into TTA ascending or PTA
ascending/reversed variants; shared geometry owns contextual addressing and radial mirror parity.
`unification.tiles` parses each strict `TILE_SIZE:TILE_STRIDE` group, while `geometry` builds the
collapsed direct-to-output tile transform. `gaussian.binary_gaussian_pass` supplies the one
constant-zero, truncate-4, threshold-at-0.5 numerical operation; mode-owned orchestration decides
whether it runs before geometry (PTA) or after fused prediction (TTA).

Grouped-view duplicate behavior deliberately follows TTA. Exact repeated tilted groups and
overlapping tilted groups that generate the same concrete signed-angle/direction view are
deduplicated. Duplicate tile groups are errors. Duplicate Cartesian tokens and repeated
assignment of a radial target remain errors because those forms are ambiguous under their
respective grammar.

## Backend boundary

`TtaScheduler` is the sole mutable owner of TTA process-inference queues, dynamic task
identities and totals, tile-result reservations, hybrid/D1 claims, backend cost estimates,
result transport, and worker accounting. Its frozen inputs and injected operations keep
lower layers unaware of orchestration, while synchronous main-thread callbacks return
completed leases to the still-inline view/tile assembly owner. The pipeline consumes an
immutable result snapshot for output metadata rather than reading live scheduler counters.

`XTA.inference_backends` contains dependency-free control-plane contracts. The
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
Unregistered backends must fail closed through `BackendRegistry` rather than falling through to
CUDA behavior.

## Multiprocessing rules

- Launch work through the sole versioned script, installed `xta` command, or
  `python -m XTA`; all three use the same strict mode-aware dispatcher. Do not load
  modules under ad-hoc aliases.
- Worker targets remain module-level importable functions in `XTA.workers`.
- Keep Torch, CuPy, TensorRT, OpenVINO, Ultralytics and future TPU imports inside their
  owning runtime paths.
- Do not pass closures from `pipeline.main()` to `spawn` children.
- Preserve explicit per-worker initialization; spawn children do not inherit resolved
  process globals.
- Large data belongs behind artifact references or the existing descriptor/path transport,
  never directly in queue or future RPC messages.

### PTA scheduler exception boundary

Normal PTA CPU process rendering uses one persistent `spawn` pool for the run. The worker target
and initializer are module-level. Run-constant settings are serialized as a picklable static
contract, CPU external-policy definitions are reloaded and identity-checked in each child, and
per-volume arrays plus phase payloads travel through named shared-memory blocks. The parent owns
membership, split, augmentation-selection and output identities, so asynchronous completion order
cannot change dataset identity. The explicit thread backend shares parent arrays and remains the
fallback when `auto` cannot create a spawn context.

Active offline external GPU augmentation is the only fork-only exception. Its external factory
cannot be assumed picklable, so a process backend with GPU policy IDs uses a fork context and is
created before source decode or CUDA-context creation. A non-fork-capable host rejects that path;
it does not silently route the policy through ordinary spawn workers. This exception does not
apply to built-in CPU geometry and is not evidence of GPU production qualification. Exactly one
persistent process owns each visible CUDA device. A bounded CPU producer pool inside that owner
renders independent full/tile items while earlier work runs on the GPU; shape-compatible items
fill multi-source policy calls subject to free-VRAM admission and deterministic OOM splitting.

## Output ownership and successful-run publication

PTA has a fresh-publication lifecycle, not resume markers. Before cleanup it rejects drive or
filesystem roots, home/workspace ancestors, any input/output containment in either direction, and
generated targets that are symlinks/junctions or overlap discovered inputs or the external policy.
A nonempty existing output must contain `.pta_v18_output.json` whose schema and resolved path own
that exact directory. Cleanup touches only the enumerated generated directories/files, then
rewrites the ownership sentinel. Requested and effective image formats remain separate: parser
aliases normalize to requested `png`, `jpg`, or `tif`, while a custom `C...S...` channel layout
always gives the writer effective multipage `tif`. Both values are manifested as
`requested_output_format` and `effective_output_format`.

The successful `manifest.json` is deliberately the last selected artifact. PTA removes any prior
generated manifest during safe cleanup and publishes a new complete manifest only after output,
optional summary/voxel reporting, temporary-work cleanup, and input-identity revalidation. TTA
first atomically replaces a prior success record with `status: in_progress`, publishes all selected
outputs, verifies source/model identities, closes runtime resources and scratch, and only then
atomically replaces that record with `status: complete`. Once PTA cleanup for a new attempt has
started, failure leaves no complete manifest for that attempt; a safety/validation failure before
cleanup may preserve the previous untouched publication and manifest. A failed TTA run retains an
in-progress record rather than a stale success claim.

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

From a source checkout, the unprivileged HGX Track-A smoke test is
``python tools/v1803_hgx_selftest.py --gpus 4`` (and ``--gpus 8`` for a full-node
allocation). It creates boundary-crossing synthetic objects and requires a byte-identical
GPU/CPU `keep_objects` result. ``--plan-only`` exercises partitioning and row packing on a
host without CUDA. ``python tools/v1803_d1_ipc_selftest.py --gpus 4`` verifies that
single-visible-device spawned ranks can export, import, NVLink-OR, and acknowledge the
same dedicated CUDA allocations used by D1 groups; repeat with eight allocated GPUs.
Wheels retain both scripts under ``share/xta/tools``. Four-GPU representative TTA
qualification completed the full D1 lifetime across model workers, partial publication,
release acknowledgement, and scheduler quiescence; the byte-exact but performance-neutral
result is recorded in the HGX experiment section above.

For v18 specifically, CPU tests cover policy/plan identity, categorical and intensity geometry,
channel/tile primitives, spawn-worker contracts, manifest/ownership safety, and the implemented
CPU-backed `RenderBatch` fan-out. A bounded, nonrepresentative GPU run qualified
basic CUDA/Torch/CuPy execution and the resident renderer: Cartesian and tilted-Cartesian fixtures
met the one-uint8 cross-backend tolerance, while optimized hardware-texture Radial fixtures showed
backend-specific sampling differences and their Torch fallbacks remained within tolerance. This
found no seam-index/mirror-assembly mismatch and does not authorize a sampling-policy change.
Remaining qualification is narrower than implementation: device-resident CUDA and direct
TensorRT-ring batches still need a canonical artifact capture boundary; retained CUDA sampling,
nvJPEG and offline external GPU policies need production-device-native goldens. External augmentation already
has policy hashing/export validation, deterministic CPU selection, child-side CPU reload checks,
deferred bundle publication and paired image/mask
invariants, but still needs representative user-policy dataset runs and confirmation that the
training loader consumes deferred replay bundles. Geometry authored inside an external policy is
outside the built-in forward-policy guarantee.

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
