# XTA architecture

XTA provides test-time augmentation (TTA), pretraining augmentation (PTA), and
label-time augmentation (LTA) for volumes. The implementation lives in the
importable `XTA` package. The versioned launcher
`GPT-6-Astra-Ultra_v21.1.0_SLURM.py`, installed `xta` command, and `python -m XTA`
all enter `XTA.cli.run()`.

This document describes implemented behavior, ownership, and operating controls.
Release narratives, benchmark results, validation receipts, and individual SLURM
runs are kept in the local workspace's ongoing, un-versioned
[experiment log](../Scratch/Data/XTA/History/Experiment_Log.md), outside the
repository. Accepted findings about behavior, ownership, operating controls, and
validation are incorporated here before superseded experiment notes are removed.
The log retains outcomes, limitations, source identities, and evidence locations.

## Execution model

| Mode | Input and purpose | Execution and publication |
| --- | --- | --- |
| TTA | A source volume and segmentation model | Render physical views and variants, infer masks, assemble completed views, project to source coordinates, fuse and postprocess the union, publish selected layers and derivatives |
| PTA | Source volumes and labels | Apply shared geometry and paired augmentation, select deterministic dataset candidates and splits, publish images, labels, and a complete dataset manifest |
| LTA | A target volume, aligned exemplar annotations, and a local SAM bundle | Propagate authoritative masks through temporal sessions and overlapping tiles, settle spatial relays, project each physical view once, publish the final native union and manifest |

Configuration is validated before mode-specific runtime dependencies load.
`XTA.__init__` is inert, and the eager package import graph is acyclic. Lower-level
modules expose explicit contracts; callbacks into higher-level owners use local
imports. Spawned functions and data types resolve through canonical module paths.

## Module ownership

All module names below are relative to `XTA`.

| Boundary | Modules and responsibilities |
| --- | --- |
| Configuration and entry points | `cli`, `config`, `tta_mode`, `pta_config`, `pta_mode`, `lta_config`, `lta_mode`: mode grammar, validated options, and deferred dispatch |
| Workspace and processes | `workspace`, `runtime`, `media`, `workers`: backing policy, telemetry, NUMA, process setup, source decoding/readiness, and TTA worker entry points |
| Forward-render contracts | `unification.contracts`, `unification.sampling`, `render_batch`: immutable requests/plans, executable sampling policies, and actual frame ownership |
| Shared planning | `unification.channels`, `unification.tiles`, `unification.views`, `unification.runtime`: channel/tile grammar and physical-view compilation |
| Run identity | `unification.context`, `unification.manifest`, `unification.tta_manifest`: launch context, artifact identity, and atomic manifests |
| Cartesian and Azimuthal geometry | `geometry`, `backprojection`, `sparse_projection`: forward raster plans, affine/seam geometry, and dense or crop-driven inverse projection |
| Radial geometry | `cylindrical_geometry`, `cylindrical_projection`, `cylindrical_cuda_projection`, `cylindrical_owner`: cylindrical shells, bounded pull plans, CUDA source upload/publication, and native shell ownership |
| Spherical geometry | `qsc`, `spherical_geometry`, `spherical_cuda`, `spherical_sampling_cuda`: QSC coordinates, shell plans, cached directions, and native rendering |
| Spherical projection | `spherical_projection`, `spherical_projection_bounds`, `spherical_projection_cpu`, `spherical_projection_cuda`, `spherical_preflight`: admission, conservative bounds, CPU/CUDA pulls, and preflight |
| Model execution | `inference`, `inference_backends`, `cuda_backend`: backend contracts, model execution, mask payloads, and resident CUDA rendering |
| TTA scheduling | `pipeline`, `tta_scheduler`, `tta_prediction`, `tta_lifecycle`: preparation, process admission, source staging, and run-resource ownership |
| TTA completion | `assembly`, `tta_terminal`, `tta_outputs`: view/tile assembly, physical-view terminal fusion, and settled-artifact teardown |
| Sparse components | `interpolation`, `topology`, `topology_runs`, `projection_queue`: interpolation, component membership/adjacency, and bounded projection handoff |
| CUDA component work | `cuda_interpolation`, `cuda_d1`, `cuda_finalization`: bridge painting/radius work, owner-GPU bitsets, and distributed finalization contracts |
| Shared filters and output | `gaussian`, `finalization`, `outputs`: binary Gaussian semantics, global filtering, NRRD/media output, summaries, and derivatives |
| Compact publication | `packed_publication`, `publication_memory`, `nrrd_spans`: row-packed payloads, retained-RAM grants/spill, and native crop-row streams |
| Replay | `component_replay`: bounded immutable component captures, geometry descriptors, checksums, and replay loading |
| PTA orchestration | `pta`, `pta_runtime`, `pta_dataset`, `pta_augmentation`, `pta_scheduler`: source planning, dataset identity, policy loading, GPU admission, and work packing |
| PTA execution | `pta_rendering`, `pta_workers`, `pta_publication`: render plans/caches, process/shared-memory ownership, and atomic image/label publication |
| LTA planning | `lta_inputs`, `lta_runtime`, `lta_scheduler`: annotation discovery, prompt ranking, physical-view/device ownership, and session admission |
| LTA model boundary | `lta_sam`, `lta_experimental`: local SAM provenance, runtime-neutral result contracts, and pinned authoritative-mask tracker integration |
| LTA propagation | `lta_propagation`, `lta_windows`, `lta_tiles`, `lta_tile_tracking`, `lta_relay_episodes`, `lta_tracklets`: temporal windows, spatial relay, lineage matching, and authoritative handoff |
| LTA execution/output | `lta_execution`, `lta_workers`, `lta_worker_adapter`, `lta_cpu`, `lta_telemetry`, `lta_rendering`, `lta_union_artifacts`, `lta_postprocessing`, `lta_outputs`: persistent GPU workers, allocation-aware CPU budgets, phase traces, sparse mask transport, relay convergence, one-time backprojection, and final publication |
| Optional accelerators | `experimental_features`, `intel_compression`, `intel_dsa`, `nvtiff_backend`: feature admission and hardware-specific lifecycle boundaries |

## Shared geometry and rendering

### Sampling and plan identity

`ForwardSamplingPolicy` declares coordinate/stage order, role-specific kernels,
boundaries, and registered CPU/CUDA implementations. Runtime code binds a backend
and data role through `require_forward_sampling()`. An absent binding is an error.
Manifests record resolved implementations through
`forward_sampling_execution_record()`.

Every built-in full-frame or tile job carries an immutable `RasterPlan`. Its
canonical record contains mode, physical-view identity, in-plane and channel
variants, output shape, optional tile layout, frozen metadata, and sampling policy.
SHA-256 digests address both the policy and plan. Mode-owned metadata participates
in identity; equivalent PTA/TTA geometry resolves to the same implementation
without requiring identical plan digests.

`RenderRequestBatch` contains logical frame addresses and plan identities.
`render_batch.RenderBatch` contains actual frames, and each item references the
same object supplied to the model. Tail repeats and seam-extension slots are
marked explicitly. After this boundary, backends perform layout, dtype, and
normalization conversion; image sinks consume the same geometry result.
The implemented image/model fan-out covers CPU-backed main-process and worker
slab sources. Device-resident execution owns its tensors through the CUDA source
and TensorRT slot contracts.
Device-resident CUDA and direct TensorRT-ring execution still lack a qualified
canonical image-artifact capture boundary.

Intensity and categorical roles bind their own sampling rules. Categorical
sampling selects nearest source voxels and preserves label values. Built-in
geometry is shared across modes; external augmentation policies own their
additional transforms and paired image/mask behavior.

Channel expansion uses TTA ascending or PTA ascending/reversed variants, with
contextual addressing and seam parity supplied by shared geometry. Tile groups
use strict `TILE_SIZE:TILE_STRIDE` parsing and collapsed direct-to-output affine
plans. Repeated concrete tilted views are deduplicated. Duplicate tile groups,
Cartesian tokens, and ambiguous Azimuthal assignments are rejected.

`gaussian.binary_gaussian_pass` supplies constant-zero padding, truncate-4
filtering, and a threshold at 0.5. PTA applies it before geometry;
TTA applies it after prediction fusion.

### View families

| Family | Native raster and contextual direction |
| --- | --- |
| Cartesian | Transverse, Sagittal, or Coronal planes; context follows the selected stack axis |
| Tilted Cartesian | Tilted planes using the shared source/affine coordinate convention |
| Azimuthal | Diameter/height slices indexed by azimuth, with half-turn symmetry and a mirrored seam |
| Radial | Circumferential arc length by axial height; context follows neighboring cylindrical radii |
| Spherical | Fixed QSC face patches; context follows neighboring spherical radii |

Coordinates use X for source columns, Y for source rows, and Z for source stack.
The working source grid and final native-shape restoration are distinct, including
virtual reconstruction of a deferred T axis. Native shell patches have their own
physical origins. Optional Tiles select samples inside those patches.
Native shell sampling rounds each reconstructed logical-T voxel to gray8 before
shell interpolation, then rounds the native plane before subsequent affines.

Tilted Azimuthal images shear the stacking coordinate within each azimuthal
plane, changing sampling and clipping without introducing a new plane normal.
The D1 route splats at the inferred angles; it does not inherit the pull
projector's angular densification. Coarser angle policies require coverage
qualification.

### Radial cylindrical shells

`--enable_radial VIEWS` selects concentric shells around the Transverse, Sagittal,
or Coronal axis. Corresponding `tilted_*` tokens compose with configured tilt
variants. `--enable_azimuthal VIEWS[:AZIMUTH_ANGLE]` selects the diameter/height
family described above.

`--radial_min_radius auto` resolves to `imgsz/(4*pi)` working voxels. The outer
radius is `(min(plane_height, plane_width)-1)/2`, centered on the source
voxel-center grid. The modeled domain is the annulus between a finite positive
minimum and that outer radius. Shells include both endpoints with radial gaps of
at most one voxel; arc-length and height sampling retain one-voxel spacing.

Each shell is covered by `imgsz` square intrinsic patches. Circumferential
sampling is periodic across the seam, the final height band overlaps, and short
source heights receive zero padding. A patch trajectory keeps its arc/height
origin as radius changes and starts when that origin first lies on a shell.
Context and interpolation clamp at radius boundaries. Repeated periodic pixels
carry trajectory provenance and contribute through ordinary mask OR.

Dense intensity coverage is defined by nonzero source interpolation taps over the
annulus. Terminal projection selects the nearest global shell before applying
trajectory offsets and visits every periodic patch occurrence. Tilted projection
uses the ideal-height validity test, sampled shear, clamping, and ties-to-even
selection. The plane plan contains immutable CSR occurrence tables constructed
in bounded source strips; plan caching has a 256 MiB budget. Concurrent requests
for one plan key share a build future. Sampled-shear metadata uses bounded batches.

PTA shell rendering requires positive `imgsz`. LTA's production view selection is
defined in its execution contract below.

### Spherical QSC shells

`--enable_spherical VIEWS` selects concentric spherical shells for TTA and fully
labeled PTA. Upright base aliases compile to one canonical QSC cube, while their
requested identities remain in the manifest. Tilted aliases share one cube per
direction and signed angle. Distinct tilt groups retain distinct identities.

Spherical tilts rotate the cube charts rigidly: positive vertical tilt rotates
about +X, and positive horizontal tilt about -Y. The center is
`((W-1)/2, (H-1)/2, (T-1)/2)` and maximum radius is `(min(T,H,W)-1)/2` in working
coordinates. `--spherical_min_radius auto` independently resolves to
`imgsz/(4*pi)`. Positive finite radii cover the declared annulus, include both
endpoints, and have gaps no larger than one voxel.

The chart is the equal-area O'Neill-Laubscher Quadrilateralized Spherical Cube
used by [PROJ](https://proj.org/en/stable/operations/projections/qsc.html).
`qsc` implements the unit-sphere mapping; `spherical_geometry` applies the
source-volume pose. Every face uses an endpoint-inclusive grid sized for outer
radius R: the interval count is the smallest even `n >= 3*sqrt(R*(R+1/2))`.
Fixed `imgsz` patches cover its `n+1` nodes per axis, overlap at the final patch,
and center/zero-pad smaller faces. All radii share the same face lattice and
patch origins. Incident closed faces, edges, corners, and overlapping patches
contribute through mask OR.

Spherical cleanup and interpolation operate within each fixed patch trajectory
before source-space OR. Whole-shell stitching and neighboring-face halos are not
part of this path; model quality across patch and face boundaries is a separate
qualification from coordinate coverage.

The QSC inverse has conservative Euclidean Lipschitz bound 5/3. Combined with the
radius spacing, the lattice places each working-voxel center in the annulus
within squared distance `281/324 < 1` of a native sample. This establishes
positive trilinear input weight; categorical sampling follows its separate
nearest-neighbor policy.

Native CUDA rendering reuses direction/validity plans across radii in a bounded
256 MiB cache. FP64 plans are assembled in 64-row host strips and uploaded as
complete direction/validity arrays when they fit. Oversized entries use bounded
strips. Cache identities include geometry and precision; stream ownership keeps
entries alive until their CUDA readers finish.

Source projection evaluates FP64 QSC coordinates, global nearest-shell/QSC-pixel
selection, and closed-face membership before applying patch offsets. Shell
midpoint ties choose the inward shell; pixel ties use global round-to-even.
Incident faces contribute independently by OR. Conservative bounds come from the
rotated closed face cone and contributing radius interval; nonempty-shell
metadata can tighten that interval. Outward rounding preserves face boundaries,
radius ties, and restored voxel centers. CPU and CUDA paths skip proven empty
ranges while retaining categorical selection arithmetic.

## TTA inference, scheduling, and completion

### Scheduler and process boundaries

`TtaScheduler` owns mutable process-inference queues, task identities and totals,
reservations, hybrid/D1 claims, backend estimates, result transport, worker
accounting, and liveness checks. Synchronous callbacks return completed leases
to the assembly owner. Output metadata consumes an immutable scheduler snapshot.

An `ExecutionTarget` is one scheduler-visible execution unit: a local CUDA GPU or
a socket-local OpenVINO process. `DispatchLease` separates logical and executed
slice counts. `ArtifactRef` carries a location rather than array contents.
`BackendRegistry` requires an explicit registered backend for every lookup.

Worker entry points are module-level functions in their owning worker modules.
Spawn children receive explicit initialization and picklable contracts. Bulk data
travels through shared-memory descriptors and artifact paths. Torch, CuPy,
TensorRT, OpenVINO, Ultralytics, and SAM initialize inside their runtime owners.

OpenVINO request callbacks enqueue indexed completions. A bounded consumer pool,
sized to useful request concurrency, decodes outputs and writes destinations.
Destination slices and aggregate statistics have separate locks. Failure drains
callbacks and requests unconditionally; submission order determines which
competing failure is reported.

### Mask composition and precision

Backend execution precision and output-binding dtype are separate contracts.
TensorRT buffers use the engine's declared FP16/FP32 bindings. Generic direct
inference compacts accepted detections on device and selects scalar or tiled
mask-union kernels by layout and dtype.

The FP32 C32 tiled kernel keeps a covered pixel's prototype vector in registers,
checks bounding boxes before dot products, and reuses that vector across
detections. FP16 tiled kernels load half operands and preserve FP32 accumulation
order. Boxes, confidence, and maximum-logit planes use FP32. Union reduction
avoids a dense detection-by-image mask stack. Output sampling applies bilinear
upsampling, thresholding, native warp, and cleanup in the selected policy order.
`YOLO_TTA_DIRECT_TILED_PROTO_UNION=0` selects scalar composition.

Optional tiled-workspace allocation failure disables further tiled allocation
probes for that task. Receipts report actual shapes, dtypes, selected kernels,
and fallbacks.

FP32 bindings can contain FP16-representable values. An exact FP16 operand
round-trip on saved tensors does not qualify a different binding dtype, CUDA
accumulation order, rebuilt engine, or end-to-end output policy.

### Native TensorRT pipeline

`YOLO_TTA_NATIVE_TRT_RING=1` enables the resident batch-one Radial/Spherical
pipeline. Two persistent slots own input/head/prototype tensors, TensorRT
contexts, inference graphs, and inference/post streams. The rendering engine
fills static input slots on its render stream with the generic normalization and
requested FP16 rounding sequence before conversion to the actual input dtype.

Native postprocessing composes logits in FP32, performs network-grid bilinear
upsampling and thresholding, and applies nearest native warp. Radial owner cleanup
and Spherical parent cleanup own morphology for these views. Completed rows,
including empty masks, fulfill radius-coverage accounting.

Inference compatibility depends on engine/input geometry. A drained policy
transition retains compatible TensorRT contexts, binding tensors, and inference
graphs while rebuilding postprocessing state. Post graphs are keyed by destination,
policy, affine, shape, and bbox requirements. Setup/capability refusal before data
consumption uses the generic route. Failures after native inference starts abort
that task without replaying model work.

### Parent memory admission and Spherical retirement

Admission accounts for committed native parent capacity before dispatch, including
shared slice-write accumulators and retained Radial publication grants. Active
parents receive completion priority. View and byte credits bound preparation,
inference, completed canvases, and projection independently. Explicit file-result
mode validates aggregate canvas capacity against the dense limit.
Canvas credits bound committed canvas capacity; transient rendering/projection
allocations and total process RSS require additional headroom.

Cacheable Spherical direct-union tasks prefer a worker's last queued parent or a
distinct newly admitted parent. This placement hint is subordinate to ownership,
memory limits, hybrid/D1 rules, and work stealing.

Live Spherical projectors can request an exclusive worker-GPU retirement lease.
Completed-canvas pressure arms at 75% of the dense window and clears at 62.5%.
Continuously refreshed demand becomes eligible for age-based admission after
30 seconds. Workers finish queued inference before lending the GPU; stage leases,
auxiliary owners, memory admission, and failure fences remain authoritative.

Eligible requests are FIFO. A drained worker serves at most two projectors before
returning to inference, with a two-second successor handoff window. Unrefreshed
requests expire after 30 seconds; failed admission applies a ten-second cooldown.
CPU projection continues while awaiting admission and rechecks after eight
published slices or one second. Promotion cancels unpublished CPU work at bounded
pull-chunk boundaries and joins all readers before releasing borrowed sources.

### Streaming terminal fusion

A runtime view becomes terminal after its full-frame/tile continuation retires.
When all variants of a physical view are terminal, ownership leaves the inference
registries, variants are OR-collapsed, and terminal projection can overlap
inference on other views. Completed physical views feed a path-backed,
single-writer source-space union reducer.

Equal-geometry sparse component layers are ORed before one restoration per output
slice. A single dense handoff credit bounds waiting source-sized volumes.
Finalization/reducer futures participate in scheduler liveness and terminal
coverage checks. Global centerline and smoothing stages consume the complete
union.

After inference and D1 ownership drain, CUDA workers fence and release their
rendering source, textures, model, graphs, and allocator assets. An acknowledged
barrier precedes post-inference GPU admission. Validation refusal retains those
assets; failure after release begins is fatal. Successful retirement closes
further inference admission while permitting auxiliary mask work.

## Source projection and compact publication

### Native Radial owners

Eligible GPU-only, angle-zero, batch-one gray Radial runs use one native owner per
view inside the active D1 scheduling/publication envelope when confidence/radius
thresholds are zero and interpolation, Tiles, and native debug exports are absent.
Other configurations use completed-view parent projection.
`YOLO_TTA_RADIAL_OWNER` controls this admission.

The inference owner consumes complete native radius chunks, cleans each complete
radius union, and gathers its owned source positions into a persistent uint32
bitset. Native cleanup fills four-connected background components that do not
reach the foreground crop boundary. The gather uses the cylindrical pull plan,
including periodic occurrences, ideal-height validity, FP64 sampled shear,
clamping, and ties-to-even rounding.

Cleanup and projection are preflighted before model loading. Admission reserves
bitset, geometry, reusable label buffers, and VRAM headroom. Coverage accounting
checks both task results and consumed radii. Active owners block worker
retirement, and failed fences retain device ownership through the fatal path.
Completed bitsets enter bounded asynchronous CVOL publication with native-shell
provenance.

### CUDA source upload

Radial and Spherical completed-view CUDA projectors share cropped source storage
and upload machinery. Valid foreground boxes select uint8 rectangles from strided
native masks, with per-shell uint64 offsets. Empty shells contribute no payload;
source layouts without useful crop bounds use dense upload. Geometry continues to
use global coordinates and the same bbox guards.

Cropped packing writes directly into a bounded pinned stage, capped at 64 MiB.
The compiled packer uses validated uint64 byte indices. When the payload exceeds
the stage and compiled packing is available, the default cropped-upload pipeline
splits that stage into two lanes. CPU packing can overlap the previous lane's
asynchronous upload; an event fences each lane before reuse, and a final stream
fence settles the transaction. Both lanes share the original staging budget and
one device source allocation. Smaller transfers and the NumPy packer use the
serial path. `YOLO_TTA_CROPPED_UPLOAD_PIPELINE=0` selects serial staging.

Source setup reports logical bytes, actual H2D bytes, storage layout, packing,
upload, geometry, and preflight separately. A failure after upload begins fences
owned resources before propagating.

### Encoded projection output

An exclusive CUDA stage lease covers upload, computation, callbacks, and cleanup.
Admission checks actual free VRAM for source, geometry, bounded buffers, and
reserve. Capacity refusal or settled setup failure selects CPU before publication.
A late failure aborts the sink; an un-fenceable stream retains its owners through
the fatal path.

CPU/CUDA projection publishes ordered bounded blocks of tight slice crops,
foreground counts, and raw uint8 or little-endian row-packed payloads.
Analytically empty ranges produce validated empty records. CUDA computes crop
metadata and packs payloads on device; sink-only Spherical CPU workers compute
those records before handing them to the publication thread. Generic dense
callbacks use their dense interface.

The writer validates order, bounds, counts, layout, and padding before appending
encoded records without repeating full-plane bbox scans or packing. Encoded work
has bounded metadata/payload reservations and a 4,096-slice block limit.
Publication is ordered and exactly once across CPU-to-GPU promotion.
A sink failure aborts the writer; completed output is not replayed.

Spherical mathematical preflight checks complete planes through 65,536 pixels.
Larger probes use deterministic windows totaling at most 32,768 pixels, including
image, ROI, and foreground edges and interior samples. Full GPU codec checks
validate payload handling. `YOLO_TTA_SPHERICAL_FULL_MATH_PREFLIGHT=1` requests the
exhaustive mathematical check in bounded CPU chunks.

`packed_publication` scans source bitset words for exact slice bounds and counts,
then emits cropped row-packed bytes. Compiled and bounded NumPy implementations
share the same payload contract. Windows file descriptors use binary mode.

## Sparse components and interpolation

Full-frame, zero-angle Cartesian interpolation components preserve sparse storage.
Transverse components reuse immutable stores; Sagittal and Coronal components
transpose into packed orthogonal stores. Azimuthal and tilted-Azimuthal projection
visits foreground crops through the discrete inverse ownership map. Ordinary
tilted Cartesian components use the dense projection backend.

Eligible D1 continuations publish complete component references alongside their
independently published source-space base. Membership export uses validated paste
bounds and counts, with full-slice export when the same payload cannot be
established from a crop. Tiles, retained workspaces, and incomplete component
coverage use ordinary continuation.

`projection_queue` accepts immutable components independently of preparation.
Pending input bytes and estimated active scratch have separate limits, and one
oversized job can run alone. Parent dependencies remain live until every future
settles. Failure wakes blocked producers before executor teardown.

Sparse labeling uses foreground bounds to select cropped CPU work at low
coverage. Topology adjacency intersects equal-label row runs where useful and
uses bounded pair deduplication. Small or highly fragmented windows use pixel
adjacency. Final fusion ORs sparse crops directly when only temporal restoration
is required.

Interpolation endpoints use odd, centered rectangular canvases with background
margins. Endpoint travel is applied by the world-coordinate painter. The CPU
min-radius evaluator can certify acceptance for a positive threshold from a
common foreground disk, guarded by a floating-point margin; other plans receive
the full section-radius scan.
The certificate is an acceptance lower bound; requests without a positive rejection
threshold still compute the full radius. Shape-dependent floating-point EDT rounding
can change boundary voxels, so the rectangular implementation does not promise
bit-identical output.

Bridge membership excludes the immutable pre-pass foreground before source
projection. Every combination in one pass uses that same pre-pass domain, and
accepted changes become input to later passes. Source-space subtraction is not
equivalent because projection can map several view pixels onto one source voxel.

CUDA bridge painting borrows an already-warm worker. Its first nonempty bounded
batch runs on CPU and CUDA; CUDA is retained when at least 5% faster. Painting is
OR-idempotent, enabling failed-batch CPU replay. A lease owns a bounded pool of
non-default streams, four by default, and retains touched cache entries until
stream completion. Pinned nonblocking result copies complete before host crops
are committed. Renderer locks protect cache metadata and enqueue order.

The CUDA radius evaluator is separately opt-in. Radius failure can return radius
work to CPU while a healthy painter continues. Required-CUDA mode makes admission
or execution failure fatal. Dedicated interpolation processes use explicit
initialization; context creation in those processes or the main process is
separately controlled.

## PTA dataset execution

PTA accepts one resolved configuration for discovery, preprocessing, geometry,
augmentation, dataset planning, and publication. The parent owns candidate
membership, augmentation versions, train/validation splitting, and output
identity, so asynchronous completion cannot change the dataset definition.

CPU process rendering uses one persistent spawn pool with module-level targets
and a picklable static contract. Children reload and verify external CPU policy
identity. Per-volume arrays and phase payloads use named shared memory. An
explicit thread backend shares parent arrays and is the automatic fallback when
a spawn context cannot be created.

Active offline external GPU augmentation uses a fork pool created before source
decode or CUDA initialization. It requires a fork-capable host. One persistent
process owns each visible CUDA device; bounded CPU producers prepare compatible
full-frame/tile items while earlier GPU work runs. VRAM admission and deterministic
OOM splitting bound policy batches. GPU example policies implement separable
Gaussian filtering for blur and elastic-field smoothing.

PTA partial-label and encoded-gap input paths use Cartesian labeling. Fully
labeled data binds the shared supported forward geometry, including native shell
families. External policies carry identity and deterministic selection metadata;
deferred replay bundles are published as explicit dataset artifacts.

## LTA propagation and SAM ownership

Production LTA executes native Transverse, angle-zero, overlapping 1008-pixel
tiles. Aligned exemplar indexes address decoded target frames. One persistent
spawned worker owns one model per selected GPU. A physical-view owner retains its
immutable rendered cache and sole backprojection ownership. Idle devices can
assist unopened sessions using that cache; a live session stays on its original
device. Results commit in plan order.

Production helpers take the earliest ready windows. Each selected device keeps
one task slot for a single existing SAM window of at most 30 frames. Verified
boundary seeds unlock the next window; a center window unlocks backward and
forward continuations independently. A live tracker session stays on one GPU;
the next fresh session can run on another. Blocked dependencies stay outside
ready queues. An empty boundary cancels only its dependent branch. Generation
counts and an ordered commit cursor avoid rescanning the entire work history.

Workers retain at most one window's dense union and publish only nonempty,
tightly cropped frames as little-endian row-packed bits. Omitted frames mean
zero. Hashing and consumption scale with stored support, not the full source
depth. The coordinator verifies the packet, ORs only its indexed crops into the
private view, removes it, and admits more work. Compact audit records still
commit in plan order. Cross-window relay episodes merge at the original chain
boundary; window seams do not create additional spatial seeds. Complete relay
generation fan-in remains necessary because seeding merged arrivals and unioning
separately propagated arrivals are not equivalent operations.

Input discovery reports its active scan, probe, and exemplar identity stages.
Video frame counts use declared metadata when available; FFV1 inputs without a
declared count use a packet scan, while other codecs retain a decoded-frame
fallback. Frame counts are never estimated from duration and rate. LTA verifies
the count against EOF during source-cache decoding, rejecting both missing and
extra frames, with concurrent stderr draining to prevent pipe deadlocks.

Preflight checks scratch and output filesystem capacity,
combining reservations when they share a filesystem, and includes the unfiltered,
requested filter-stage, and final NRRDs. Relay growth and optional media remain
additional storage consumers.

CPU budgets intersect process affinity with Slurm CPU limits and divide native
threads across selected GPU workers, capped at four per worker while respecting
an inherited lower limit. Child environment limits apply before adapter imports;
Torch and OpenCV limits are set before model construction. Parent native pools
are scoped to one thread while explicit LTA CPU parallelism uses the effective
allocation. Original parent settings are restored on exit. An explicitly empty
`--temp` value is rejected so an unset scratch variable cannot select output
storage unintentionally.
The unique `lta_<run-id>` scratch directory is created after discovery, planning,
and capacity preflight; its creation and resolved path are printed immediately.

`lta_execution_identity.json` records the actual source fingerprint and execution
contract, including the resolved scratch directory. Per-process JSONL traces under `lta_diagnostics` identify decode,
planning, startup, queue waits, rendering, SAM sessions, sparse reduction, and
relay work. The coordinator reports ready, blocked, and active window counts;
an impossible blocked graph fails with diagnostics instead of spinning.
`tools/lta_trace_summary.py` summarizes these host phases, including unfinished
phases in an ongoing or interrupted run. Host phase time is not CUDA kernel time.

Sessions span at most 30 frames and admit at most 128 objects. Seed groups are
partitioned deterministically. Authoritative masks, temporal dogfood, and lineage
identity travel through explicit session contracts. Tracker confidence uses the
sigmoid framewise score; removal sentinels represent bookkeeping. Filled
predictions stream into bounded window unions and bit-packed relay reducers, while
boundary dogfood and compact audit state remain resident.

The pinned single-rank SAM 3.1 mask tracker prepares visual features without
running unused grounding detection. `lta_tracker_features` preserves both tracker
necks, all six FPN levels, positional encodings, the original BF16 conversion
before decoder projections, and the inherited autocast context. Unsupported
custom or distributed model layouts retain their original preparation path;
failures inside the supported path propagate. Per-session receipts and the final
worker audit count direct feature preparations and fallbacks. Installed SAM source
remains unchanged. Mask batches and score/sentinel/finite-check vectors cross to
the CPU together, while masks retain independent ownership. LTA uses TTA's exact
foreground-bbox plus halo hole fill to reduce per-instance CPU work.

Every authoritative anchor starts a separate chain across the full physical-view
frame range. Fixed-size center, backward, and forward windows preserve lineages
across later partially annotated anchors; another annotation does not terminate
an existing object. Initial chains combine by recall union. A temporal branch
stops when its shared boundary has no eligible seed, so recovery through an empty
dogfood boundary remains unsupported. Predictions must match their originating
sequence, session, and seed raster dimensions before entering any reducer.

Initial, relay, and temporal-dogfood seed groups use deterministic first-fit
partitioning over their aggregate shared-pixel domain. Every mask must retain at
least 95% exclusive support within the 128-object session limit; the worker repeats
admission at each window boundary. Conflicting lineages remain distinct and retain
complete seeds in separate sessions. Mask overlap alone does not establish object
identity.

Adjacent-anchor identity matching and confidence-based handoff are available in
`lta_tracklets` and the `tools/lta_tracklet_pair.py` diagnostic. Conservative
one-to-one assignment preserves unmatched tracklets; split/merge hypotheses are
audit evidence. The diagnostic restores annotated foreground by OR so partially
labeled anchor slices retain unmatched objects. Production currently uses recall
union and records that cross-anchor identity reconciliation is not applied.
Connecting handoff to production requires retaining per-instance masks and
probabilities before union and planning spatial relays from the reconciled masks.

SAM seed previews exclude pixels shared by simultaneously injected masks.
Production validates each representable exclusive mask and its union exactly;
non-shared erosion, expansion, identity changes, and insufficient exclusive support
fail. Later tracked prompt-frame IoU is diagnostic: production replaces that
preview with the exact filled seed and restores authoritative foreground after
terminal filters.

Edge-pinned tile grids use actual eight-neighbor overlap. Each contiguous overlap
episode generates forward and backward relays. Masks are rebased in global view
coordinates. Event identity includes lineage, destination, frame, and temporal
direction: repeated events suppress duplicate work, while new foreground advances
mask revisions. A complete polygon uses its strongest complete-mask tile; larger
polygons keep all required authoritative fragments.

A bounded breadth-first fixed point settles cross-tile growth. A safety-cap hit
with pending growth fails publication. Completed predictions collapse in
physical-view space, receive final 2-D hole filling, backproject once, and enter
ordered native-union postprocessing. Immutable hard-positive foreground is
restored after destructive filters. LTA always saves `Global_union_before_postprocessing`,
one `Global_after_<filter>` checkpoint for each requested postprocessing operation,
and `Global_final_output` as NRRDs, even without `--save nrrd`. Full independent
checkpoints use TTA's `checkpoint`/`select` metadata. The `after_keep_objects`
checkpoint shows the exact filtered result; final output additionally restores
original hard-positive annotations, which can reintroduce disconnected components.
Checkpoint writes are atomic and durable after each stage. Their sidecar preserves
filter settings and hashes if a later operation fails. Source/model identities are
validated before the first checkpoint and again before final publication; only the
complete run manifest marks overall success. Explicit empty annotations are
audited as known background.

SAM checkpoints and BPE assets are local. Workers verify the pinned distribution,
commit/source tree, and BPE identity before model construction. The SAM 3.1
adapter loads one memory-mapped assembled state, audits keys/device/dtype
placement, and chunks large host-to-device copies. Scoped attention fallbacks and
constructor adapters restore upstream functions when their scopes close.

The reusable builder defaults to CPU construction and FP32 storage. Production
profiles use meta construction with compilation and warmup disabled:

| Profile | Admission | Storage and batching |
| --- | --- | --- |
| `h100` | Hopper or newer compute capability | FP32 weights and ordinary batches |
| `egpu` | RTX 4090-class device | `bfloat16_egpu` weights, FP32 decoder FFN linear layers, and constrained batches |

Automatic selection chooses the appropriate profile and applies its device
validation. The BF16 profile's decoder FFN exceptions are explicit audited dtype
boundaries. Diagnostics for box, composite, and point prompting remain separate
from authoritative-mask production propagation.

## Storage, output, and completion transactions

### Workspace and memory backing

Ephemeral scratch uses shared mappings without synchronous writeback. Raw CVOL
payloads are pathname-backed; source/result shared buffers have explicit
allocator ownership and release accounting. Linux mount classification uses the
kernel mount ID of an opened path, or its nearest existing ancestor. Storage
medium and persistence are separate properties: job-local temporary paths are
released with the allocation.

Linux native shell payloads can use parent-owned memfds under a run-wide RAM
plan. Every selected future layer is charged its worst-case packed size before
dispatch. Admission considers physical/cgroup headroom without swap, final union,
topology labels, publication credits, codec windows, mirrors, and spools, and
leaves half the remaining headroom unused. Configured retained-payload and
anonymous-workspace caps apply.

Parent descriptors survive worker exit until consumer retirement. Between
producer callbacks, headroom pressure or grant exhaustion spills unfinished
payloads to disk and releases RAM pages. A failed spill preserves the original
payload and propagates. Shared-mapping cache identity uses the absolute logical
layer path without resolving its payload symlink, so distinct memfd-backed layers
keep distinct mappings through retirement.

### Output encoding and ownership

Native CVOL NRRDs using software member codecs stream nonempty crop row bands
through the configured codec. Empty rows/slices use reusable gzip zero members
at bounded power-of-two sizes through 1 MiB. Completion queues are bounded, and
sparse mirror observers receive complete crops. Restored geometry and dense
observers use their corresponding assembly paths.

Optional Intel QAT/QATzip and IAA/QPL extensions provide hardware gzip with
explicit admission, framing, and failure checks. DSA provides opt-in Linux idxd
workspace copy. Build/provisioning and codec-specific contracts are documented in
[native/README.md](native/README.md), [QAT notes](native/README_QAT.md), and
[QPL notes](native/README_QPL.md).

PTA validates input/output containment, generated-target ownership, and link
safety before fresh-publication cleanup. A nonempty output directory requires
its matching `.pta_v18_output.json` ownership sentinel. Cleanup touches enumerated
generated artifacts. Requested `png`, `jpg`, or `tif` remains distinct from the
effective format: custom channel layouts publish multipage TIFF, and both values
are recorded in the manifest.

A complete manifest is the final publication commit marker after selected
outputs, input/model identity checks, resource closure, and scratch cleanup.
TTA first atomically publishes `status: in_progress` and replaces it with
`status: complete` after success. PTA removes the prior generated manifest when
safe cleanup begins. LTA releases temporary ownership before complete publication.
Failure therefore cannot present an incomplete attempt as a completed run.

## Runtime controls

Controls below request a path; shape, backend, resource, and ownership guards
still decide admission. Local component overrides take precedence over bundle
settings. Actual execution is recorded separately from requested policy.

### Geometry and projection

| Control | Default | Effect |
| --- | --- | --- |
| `YOLO_TTA_FAST_GEOMETRY` | Off | Requests Spherical FP32 sampling, compiled Spherical CPU pull, and Radial column reuse |
| `YOLO_TTA_GPU_SPHERICAL_FP32` | Bundle value | FP32/FMA native intensity sampling when every native/logical source axis is at most 4,096 |
| `YOLO_TTA_CPU_SPHERICAL_COMPILED` | Bundle value | Numba scalar FP64 pull with `fastmath=False` |
| `YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY` | Bundle value | Compute FP64 column geometry once and reuse it across rows |
| `YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL`, `YOLO_TTA_GPU_RADIAL_NATIVE_KERNEL` | On | Native CUDA input samplers with resident Torch fallback |
| `YOLO_TTA_GPU_SPHERICAL_BACKPROJECT`, `YOLO_TTA_GPU_RADIAL_BACKPROJECT` | On | Completed-view CUDA projection admission |
| `YOLO_TTA_CPU_SPHERICAL_COMPACT` | On | Encoded crop publication from sink-only CPU projection |
| `YOLO_TTA_CROPPED_UPLOAD_PIPELINE` | On | Two-lane staging for eligible large compiled crop uploads |
| `YOLO_TTA_RADIAL_OWNER` | On | Eligible native Radial cleanup/source-bitset ownership |
| `YOLO_TTA_GPU_SPHERICAL_LOCALITY` | On | Parent-local placement for cacheable Spherical tasks |
| `YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT` | On | Pressure/age retirement lease lane |
| `YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT` | On | Age admission within that lane |
| `YOLO_TTA_SPHERICAL_FULL_MATH_PREFLIGHT` | Off | Expanded Spherical mathematical preflight |
| `YOLO_TTA_DIRECT_TILED_PROTO_UNION` | On | Layout-eligible tiled mask composition |
| `YOLO_TTA_NATIVE_TRT_RING` | Off | Resident Radial/Spherical TensorRT slots and postprocessing |

The Spherical FP32 policy permits an absolute gray8 qualification tolerance of
two; generic CUDA keeps its tolerance of one. Categorical sampling and 64-bit
source addressing preserve their own contracts. Requested precision participates
in policy/plan identity. This tolerance is a validation policy, not a runtime
pixel comparison or a universal input-error bound.

The fast-geometry bundle selects Radial column reuse at at least 256 active rows,
256 columns, and 262,144 active pixels. Explicit component selection also admits
small cases. Optional geometry-allocation or compiler setup refusal uses the
reference route before publication starts.

### Sparse publication and interpolation

| Control | Default | Effect |
| --- | --- | --- |
| `YOLO_TTA_PACKED_OWNER_PUBLICATION` | On | Row-packed source-bitset CVOL publication |
| `YOLO_TTA_NRRD_CROP_ROW_SPANS` | On | Native software-codec crop-row streaming |
| `YOLO_TTA_PUBLICATION_RAM` | On | Eligible Linux planned retained-payload RAM tier |
| `YOLO_TTA_PUBLICATION_RAM_GIB` | No additional positive cap | Caps retained payloads when set above zero |
| `YOLO_TTA_TOPOLOGY_RUN_ADJACENCY` | On | Bounded equal-label run intersection |
| `YOLO_TTA_GPU_INTERPOLATION` | On | Warm-worker CUDA bridge-painting admission |
| `YOLO_TTA_GPU_INTERPOLATION_RENDER_AUTOTUNE` | On | CPU/CUDA first-batch comparison |
| `YOLO_TTA_GPU_INTERPOLATION_RADIUS` | Off | CUDA radius evaluator |
| `YOLO_TTA_GPU_INTERPOLATION_REQUIRED` | Off | Require CUDA and make admission/execution failure fatal |
| `YOLO_TTA_GPU_INTERPOLATION_STREAMS` | 4 | Non-default streams per lease |
| `YOLO_TTA_GPU_INTERPOLATION_RESERVE_MIB` | 1024 | Free-VRAM reserve withheld from cache sizing |
| `YOLO_TTA_GPU_INTERPOLATION_CACHE_MIB` | 1024 | Retained SDF/section payload cap |
| `YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT` | Off | Permit a dedicated interpolation process to create CUDA context |
| `YOLO_TTA_GPU_INTERPOLATION_MAIN_PROCESS` | Off | Permit main-process context creation together with the preceding control |

Interpolation's logical cache excludes transient library workspaces and allocator
pool blocks, which are released at lease closure. The global interpolation-pass
limit is one by default. Radius, painting, transfer, lock wait, cache eviction,
and fallback are reported separately.

### Optional multi-GPU transactions

`YOLO_TTA_D1_OWNER_GROUPS=1` admits deterministic slice coverage for one eligible
D1 parent across idle CUDA workers. The scheduler atomically reserves the group
before dispatch and uses `YOLO_TTA_D1_OWNER_GROUP_SIZE` to cap participants at the
visible device count. Participants retain dedicated IPC-exportable partial
bitsets until CUDA-IPC/NVLink reduction or bounded host recovery is acknowledged,
then explicitly acknowledge release. Nonparticipants continue view-level work;
one-owner execution is the admission fallback.
Participant release acknowledgments gate both worker reuse and scheduler
quiescence.

`YOLO_TTA_GPU_RESIDENT_TAIL=1` uploads a settled host union into contiguous
job-visible Z shards. Bounded device CCL blocks retain exact 26-connected labels;
compact equivalence pairs cross block/shard boundaries. CPU union-find resolves
area/boundary metadata, and the filtered candidate commits only after every GPU
succeeds. Ordinary failure leaves the host union available for CPU `keep_objects`;
`YOLO_TTA_GPU_RESIDENT_TAIL_REQUIRED=1` makes failure fatal.
Global top-N selection follows cross-shard component merging. An equal-area tie
across the keep/drop cutoff falls back to CPU to preserve its ordering; required
GPU mode raises instead. Optional-mode resident topology failure restarts from
the intact host authority; it does not continue a partial graph on fewer devices.

Both controls default off. `DistributedBinaryArtifact` defines the common
host-volume, source-bitset, and resident-shard contract. Each transaction uses
only selected job-visible devices and has explicit ownership through reduction,
commit, and retirement.
The common artifact interface does not itself implement contributor-to-resident
final-union ingestion or multi-GPU interpolation; those remain separate designs.

## Diagnostics and validation

`YOLO_TTA_TASK_TRACE=1` records bounded host task boundaries.
`YOLO_TTA_TELEMETRY_DIR` selects per-run persistent telemetry and takes precedence
over the single-path setting. Trace records identify dispatch, dequeue, compute,
publication, transport, receipts, and exclusive stage leases.
`tools/analyze_pipeline_trace.py` joins process streams and reports missing or
ambiguous boundaries. Host intervals and concurrent stage sums are interpreted
separately from CUDA-event kernel measurements and end-to-end walltime.

`--capture_component_replay PERSISTENT_DIR` records bounded immutable view-native
Azimuthal components, geometry, and checksums. Defaults select one component from
each of three vertical +30-degree tilted views within a 4 GiB input budget.
`--capture_component_views` and `--capture_component_limit` control selection.
`tools/replay_component_projection.py` compares decoded reference/sparse output
in fresh CPU processes without rerunning inference. `--cuda-reference` requests
a CUDA reference and records actual admission.

Run the source checks with the applicable dependencies installed:

```text
python -m unittest discover -s tests -v
python tools/smoke_import.py
python tools/verify_package_inventory.py
```

Run numerical corpora separately because aggregate dependency-light tests can
substitute optional dependencies with stubs:

```text
python -m unittest discover -s tests -p test_interpolation_geometry.py -v
python -m unittest discover -s tests -p test_external_augmentation_examples.py -v
```

| Validation boundary | Entry points |
| --- | --- |
| Native shell sampling/projection | Cylindrical/Spherical CUDA tests and `tools/qualify_spherical_sampling.py`, `tools/qualify_spherical_large_address.py` |
| Cropped source upload | `tools/qualify_cropped_upload_pipeline.py` and crop-upload tests |
| TensorRT ownership and masks | `tools/qualify_native_trt_lease.py`, `tools/qualify_native_trt_pipeline.py`, and native TensorRT tests |
| Sparse projection replay | `python tools/replay_component_projection.py CAPTURE_DIR --output RESULT_DIR` |
| Intel accelerators | `python tools/intel_accelerator_selftest.py --backend all` |
| Multi-GPU finalization and IPC | `tools/hgx_selftest.py`, `tools/d1_ipc_selftest.py` |
| Bounded SAM sessions | `tools/lta_gpu_smoke.py`, `tools/lta_mask_seed_smoke.py`, `tools/lta_tracklet_pair.py` |
| LTA production execution | `tools/lta_production_smoke.py` and LTA worker/execution tests; `tools/lta_full_volume.py` is a separate diagnostic |
| LTA worker profiling | `tools/lta_worker_profile.py`: heated GPU, repeated fixed cached fixtures, cProfile, utilization samples, phase times, and exact output hashes |
| LTA tracker feature parity | `tools/lta_tracker_feature_smoke.py`: compares the original and direct visual-feature paths on the same real frames, requiring exact cached tensors and no fallback |
| LTA full-depth transport and task graph | `tools/lta_host_io_profile.py`, `tools/lta_host_pipeline_smoke.py`, and `tools/lta_window_gpu_smoke.py`: full logical dimensions, spawned worker/relay protocols, and real single-GPU chain/window parity |

Hardware-backed tests require the target runtime and representative data/models.
Functional parity, numerical tolerances, and performance are separate checks.
Net voxel counts and high aggregate IoU do not establish spatial or small-component
identity; precision changes need decoded masks and topology checks. Performance
comparisons preserve the workload and requested outputs, use repeated controls,
and distinguish isolated component timing from whole-command walltime.
Keep tracing settings and telemetry storage comparable, or record their changes
as confounders in the performance comparison.
GPU benchmarks heatsoak the device before timing. Generated caches, captures,
logs, reports, builds, and release archives belong in task-specific Scratch
locations. Build intermediates created in the repository are cleaned after
validation; source, tests, tools, packaging, and this architecture remain tracked.

The package statement inventory authenticates preserved definitions and reviewed
implementation boundaries. Wheels include this document and the sole versioned
launcher, with selected replay/hardware/LTA tools under `share/xta/tools`.
