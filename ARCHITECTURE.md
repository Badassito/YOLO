# XTA architecture

`GPT-6-Astra-Ultra_v21.0.2_SLURM.py` is the sole versioned launcher. It, the
installed `xta` console script, and `python -m XTA` all dispatch
through `XTA.cli.run()`.
The implementation lives in the importable `XTA` package so spawned processes
resolve worker functions and data types through canonical module paths.

Main v20.0.3 promotes packed source publication, native NRRD crop-row streaming,
planned RAM retention with disk spill, and bounded run-based topology adjacency.
It also fixes Linux memfd cache identity. Cluster job 142621 completed all
outputs in 620.5 seconds, 44.8% less walltime than the accepted v20.0.2 run.

Main v20.0.2 promotes native shell cleanup and source-bitset projection inside
inference workers, together with bounded Radial planning and publication
improvements. Cluster job 142565 completed 205.3 seconds faster than 142543
(15.4%); the measured command and qualification limits are recorded below.

Main v20.0.1 promotes the completed v20.0.4 source from
`codex/v20-cylindrical-views`. Main v20.0.0 preserves the original cylindrical
release. The development v20.0.1 through v20.0.3 changes are included in main
v20.0.1 without separate main releases. Version references in the correction
history below retain their development branch numbering.

## Spherical view family (v21)

`--enable_spherical VIEWS` adds concentric spherical shells to TTA and fully
labeled PTA. It accepts the same six base/tilted tokens as Radial. Upright
Transverse, Sagittal and Coronal requests compile to one canonical QSC cube;
each requested alias remains recorded in the manifest. Tilted base aliases
likewise share one cube for each direction and signed angle. Different tilt
groups remain distinct, even when cube symmetry could otherwise identify them.

Spherical tilts are rigid rotations of the cube charts, not Cartesian shears.
Coordinates use X=source columns, Y=source rows, Z=source stack. A positive
vertical tilt rotates about +X; a positive horizontal tilt rotates about -Y.
This convention is independent of the requested base alias. The source sphere
stays centered at `((W-1)/2,(H-1)/2,(T-1)/2)` in the working volume, with maximum
radius `(min(T,H,W)-1)/2`. The existing final native-shape restoration remains
authoritative, including deferred T-axis reconstruction.

`--spherical_min_radius auto` independently resolves to `imgsz/(4*pi)` working
voxels. Setting `--radial_min_radius` does not affect it. Positive finite minima
are required; radii include both annular endpoints with gaps no larger than one
voxel. The excluded central core is not reconstructed by spherical masks.

The chart is the true equal-area O'Neill-Laubscher QSC used by
[PROJ](https://proj.org/en/stable/operations/projections/qsc.html), rather than
the approximate COBE cube or a gnomonic cube. `XTA/qsc.py` keeps this unit-sphere
map independent of the source-volume pose, providing an extension point for
future ellipsoid adapters without claiming generalized ellipsoid support now.

Every face uses a fixed endpoint-inclusive grid sized for the outer radius R:
choose the smallest even interval count `n >= 3*sqrt(R*(R+1/2))`, then sample
`n+1` nodes across each face axis. Fixed `imgsz` square patches cover that grid,
overlapping the final patch where necessary; small faces are centered with zero
padding. These intrinsic patches are not Tiles. Optional Tiles remain samples
inside a patch, and channels/interpolation clamp radius within the same fixed
face/patch trajectory. Faces, edges and corners use consistent QSC frames;
incident closed faces and overlapping patches contribute by ordinary mask OR.

The density guarantee is analytical: the QSC inverse has conservative Euclidean
Lipschitz bound 5/3. The chosen grid and radial spacing put every working-voxel
center in the annulus within squared distance at most `281/324 < 1` of a native
sample, so it has positive trilinear input weight. Tests enumerate those actual
taps on odd/even/thin volumes and rotated cubes. This is a native intensity
sampling guarantee; categorical labels remain nearest-neighbor and later TTA
affines follow the existing forward sampling policy. Inner shells deliberately
oversample relative to the outer sphere. A future maximum-spacing control in
working voxels could jointly set radial and face spacing; v21 exposes no sparse
density setting.

CUDA native rendering caches bounded float64 QSC directions and preserves
logical-T gray8 rounding before affine transforms. Source backprojection uses
direct float64 QSC geometry on CUDA, followed by the established compact
raw/packbits publication contract. The bounded CPU reference remains available
with `YOLO_TTA_GPU_SPHERICAL_BACKPROJECT=0`, or when device admission is unavailable.
The native renderer has a resident Torch fallback and can be opted out with
`YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL=0`. Spherical views do not enter legacy
D1/Radial owner kernels; completed-view CUDA admission is described below.

The shared LTA planner and low-level renderers record spherical geometry, while
LTA production execution retains its existing single-Transverse restriction and
rejects spherical requests explicitly. PTA partial-label and encoded-gap paths
retain their Cartesian-only labeling policy.

Functional CUDA checks cover native rendering, native-T restoration, direct
backprojection over hundreds of geometry cases, compact CVOL publication, and
uncropped source addresses beyond 4 GiB. `tools/qualify_spherical_large_address.py`
records the latter without benchmarking. Generated v21 evidence and release
artifacts live under `Scratch/Experiments/Spherical`, outside the repository.

### Spherical memory admission and retirement (v21.0.1)

Cluster job 142751 ended in a SLURM host-memory `oom_kill`. All 120 Spherical
parents had opened native `(1212,3072,3072)` uint8 accumulators before any
Spherical view completed: 1,278.3 GiB of committed canvas capacity against
975.1 GiB of initial headroom. Lazy zero-page allocation made repeated free-RAM
checks miss the outstanding commitments. Enabling D1 had globally disabled
shared unions, while Spherical correctly remained ineligible for D1. Its
file-result fallback bypassed the existing parent admission limits, and scheduling
by remaining work spread chunks across all equal-sized parents.

The D1 capability no longer disables shared unions for other families. Eligible
tasks still choose D1 first; Spherical uses disjoint shared slice writes and the
existing inference/postprocess view and byte credits. Active parents finish before
new ones are admitted, reducing QSC direction-cache churn as well as retained
memory. The scheduler no longer reads and OR-merges Spherical task result files.
Native parent capacity is also reserved before granting retained Radial publication
RAM. Explicit file-result optout rejects aggregate native canvases above the
configured dense limit with an actionable diagnostic.

These limits bound native parent canvases, not total process RSS. Transient render,
projection and output allocations still need headroom. Existing user overrides,
single-oversized-parent admission and retained-debug mode retain their exceptions.
For the normal four-worker command, a production-shaped 120-parent scheduler test
verifies at most four active inference parents and at most 256 GiB across inference
and postprocess canvases, including waiting for postprocess retirement.

A completed Spherical projection may acquire an idle worker GPU before global
inference drain when the scheduler has no admissible inference backlog. Admission
still fences queued/running inference, asset retirement and other device owners.
The lease stays exclusive even under the generic stage-overlap override. CPU
projection continues while admission is unavailable and checks again after bounded
published progress; on promotion it joins unused CPU work and resumes CUDA at the
first unpublished source slice. An admitted projection finishes to release host
memory rather than being preempted and uploading its source again. Logs record the
promotion and CPU/CUDA slice totals. Unsafe CUDA cleanup retains ownership and
fails the run instead of replaying published slices.

Spherical generic inference also preserves a compatible idle TensorRT ring across
family transitions, using the same guarded suspension as Radial. This does not
enable Spherical fused TensorRT rendering.

Local functional checks exercised actual CUDA rendering and CPU-to-CUDA retirement,
plus real `.pt` inference with split Spherical parents. All 63 resulting NRRD masks
and spatial headers match the v21.0.0 fixture exactly. TensorRT suspension has
protocol tests; this host has no TensorRT installation. The cluster's synchronized
GPU sawtooth is consistent with shared scheduling stalls and cache churn, but its
exact timing cause is not established without device/RSS traces. No local
performance claim is made. The failed command includes 145,440 additional Spherical
frames (300,805 total), so its runtime cannot be predicted from the smaller
cylindrical workload alone.

### Spherical projection and inference overhead (v21.0.2)

Job 142754 qualified v21.0.1 on the cluster: 300,805 frames, 181 NRRD outputs,
no OOM, and 1,536.9 seconds walltime. Retained native canvases stayed within
255.7 GiB. All 110 nonempty Spherical projections finished through compact CUDA,
but the first four spent 282–317 seconds including CPU progress and GPU admission.
The remaining Radial inference backlog prevented those projections from borrowing
a GPU even when the completed-canvas window filled.

When completed canvases reach 75% of the configured total dense window, a live
Spherical projector may now request one worker GPU for retirement while other
inference remains admissible. That worker finishes its queued inference before
the exclusive projection lease begins; the other workers continue inference.
Requests expire after 30 seconds and are cancelled on CPU completion. Failed
admission imposes a ten-second reservation cooldown, and asset-retirement and
auxiliary ownership fences remain authoritative. CPU progress checks GPU admission
after eight published slices or one second, whichever occurs first. No GPU is
reserved without projector demand. `YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT=0`
restores the v21.0.1 priority policy.

The Spherical CUDA projector now computes conservative source-grid bounds from
the rotated closed cube-face cone and contributing radius interval. Axes, edge
stationary points and cone corners give its directional extrema; supplied
nonempty-shell metadata can further narrow the interval using nearest-shell
midpoints. Outward rounding preserves incident faces, radius ties and restored
voxel centers. The original scalar/CUDA categorical tests still determine every
foreground byte. Active blocks retain full-size zero-filled output buffers but
launch QSC only inside those bounds. Proven empty compact slabs publish ordered
empty records without a GPU launch or crop-metadata transfer.

For the 142754 geometry, all 120 trajectories without foreground metadata require
457,443,353,472 candidate coordinate visits instead of 2,145,590,021,760 (21.32%).
This is a geometric work count, not a measured speedup; zero filling, active-block
crop scans, transfer and inference costs remain. The CPU scalar oracle is unchanged.

Cached native directions retain the same float64 QSC arithmetic in bounded
64-row host strips. A cache entry now assembles its final host arrays within the
existing 256 MiB entry budget and uploads the direction and validity arrays once
each. At 3072 square this replaces 96 small host-to-device copies and their
intermediate device copies with two uploads. Oversized entries and host allocation
failures retain the bounded strip path. Logs report cache misses, construction,
uploads and fallbacks; worker results also include warm-hit counters.

Generic direct inference uses the existing packed/tiled FP16 mask-union kernels
when their established layout guard allows it. Scalar and FP32 behavior remain
available; optional workspace OOM disables further tiled allocation probes for
that task. `YOLO_TTA_DIRECT_TILED_PROTO_UNION=0` selects the scalar path. Logs report
the selected kernel and any fallback per task. Interpolation, thresholding,
full-resolution hole filling and the Spherical TensorRT-ring exclusion are
unchanged. The ring's different proto-morphology policy is not silently applied
to Spherical masks.

Qualification includes 4,176 CPU bounds cases, an independent 27,720-direction
cone review, 432 existing CUDA projection cases, and 162 additional CUDA ROI cases
with dense/raw/packed parity. Real-model split-parent inference preserves all 63
v21.0.1 NRRD masks and spatial headers. Matched FP16 real-model runs exercise 632
tiled versus 632 scalar frames with no fallbacks and identical 63-layer outputs.
The uncropped 4.5 GiB source-address qualification also passes dense/raw/packed
checks after ROI pruning; all foreground lies beyond the 32-bit address boundary.
These local GPU runs are functional checks. Cluster TensorRT timing and overall
walltime improvements remain to be measured; the 142754 result is the baseline.

## Cylindrical view family (v20)

`--enable_azimuthal VIEWS[:AZIMUTH_ANGLE]` is the former Radial view, renamed
without changing its diameter/height sampling, half-turn symmetry, mirrored seam,
tilted composition, interpolation or projection arithmetic. Existing command lines
that request those views must use `--enable_azimuthal`. The new `--enable_radial`
has a different meaning; it is not a compatibility alias.

`--enable_radial VIEWS` selects concentric cylindrical shells for transverse,
sagittal or coronal axes. The corresponding `tilted_*` tokens compose with enabled
Tilted variants. Model axes are circumferential arc length and axial height;
radius is the slice direction. The omitted height-sliced polar family is not added.

The minimum radius is `--radial_min_radius auto` by default: `imgsz / (4*pi)` in
working source voxels. Thus one patch spans two full circumferences at the first
shell. An explicit finite positive radius can permit more wraps. Coverage is the
annulus from this minimum to the largest inscribed cylinder; the central core is
excluded. The outer radius is `(min(plane_height, plane_width)-1)/2`, centered on
the source voxel-center grid. Invalid/empty radius ranges fail explicitly.

Shells have at most one-voxel radial gaps and include both radius endpoints.
Arc-length and height pixels retain one-voxel spacing. Each shell is covered by
`imgsz`-square intrinsic patches with real periodic data beyond the 0/360 seam.
The last height band overlaps its predecessor; a source height smaller than a
patch is zero-padded. No axis is stretched to fill the square. Dense coverage
means every source voxel center in the declared annulus participates in the
intensity interpolation footprint; numerical tests enumerate actual nonzero taps.
Categorical sampling uses nearest source voxels and remains binary.

A physical patch trajectory retains its arc/height origin while stepping through
radius. Trajectories begin when their arc origin first lies on a shell. This
keeps contextual channels and interpolation on neighboring radii rather than
adjacent unrelated patches; radius boundaries clamp, with no Azimuthal mirror.
Optional `--enable_tile` ranges are extracted inside these native patches. Every
trajectory has explicit geometry/provenance in the run and render-plan manifests;
repeated periodic pixels are not labeled as independent views.

Terminal projection uses bounded source-coordinate strips and selects the nearest
global shell before applying a trajectory's radius offset. Every periodic patch
occurrence contributes by OR, retaining differing model predictions in repeated
wraps. Existing native/LQ layer publication and final source-space union apply.
New shell inference supports the generic CPU/CUDA routes; CUDA rendering uses
bounded source sampling. Eligible GPU-only, angle-zero, batch-one gray runs with
zero confidence/radius cleanup thresholds, no interpolation/tiles and no native
debug exports now project complete native shell chunks in their inference owner.
They share D1 scheduling/publication while using dedicated native cleanup and
exact cylindrical pull kernels. Other configurations retain parent projection,
with CUDA, compiled CPU and NumPy implementations. Shells never enter the old
Azimuthal sparse projector or the incompatible legacy D1 geometry kernel.

PTA requires positive `--imgsz` with Radial shells, while its existing `--imgsz 0`
native raster mode still works for the other families. LTA geometry/planning can
describe shells at its 1008-pixel model raster, but production LTA retains its
existing single-Transverse/angle-zero restriction.

### Development v20.0.1 GPU feeding correction

The native Radial CUDA renderer computes shell coordinates and source interpolation
on the GPU in one kernel per patch. It passes scalar geometry instead of uploading
the previous per-pixel index/weight maps (about 657 MiB per 3072-square frame).
Deferred source-T reconstruction, its intermediate uint8 rounding, periodic arc
coordinates, zero extension and final rounding retain the reference contract.
`YOLO_TTA_GPU_RADIAL_NATIVE_KERNEL=0` selects the retained Torch reference renderer.
A compiler/runtime fallback prints a flushed warning.

Radial tasks temporarily return the borrowed TensorRT context to ordinary inference
without destroying compatible idle ring contexts and buffers. The borrowed
context's inference graph is retired before its state changes and recaptured after
rebinding; independent context/source-keyed graphs can remain cached. Changed
source/model/binding signatures and failed synchronization still invalidate safely.
The context handoff is covered by protocol tests; actual TensorRT/H100 replay
qualification remains a cluster check.

Workers print a flushed `Inference path selected` line for each first family,
source-rendering path and result-mode combination. The fast shell kernel announces
`Radial native CUDA renderer active`. Run with `python -u` to make the remaining
SLURM stdout progress immediate as well.

Local renderer-only comparison at 3072 square, using 24 real source frames with
logical T=2911, measured 0.894–0.914 s for the reference path and 0.017–0.020 s for
the scalar CUDA kernel on the RTX 4090 Laptop GPU. All 9,437,184 output pixels
matched. This is a component measurement, not a prediction of H100 utilization
or whole-job speed. A mixed real-model smoke retained identical decoded payloads
in all 64 native/LQ NRRDs. Aggregate regression: 834 passed, 75 skipped, with
explicit CUDA and context-handoff checks run separately.

### Development v20.0.2 source projection correction

Job 142404 completed in 15,555.5 seconds, of which 14,913.9 seconds were after
inference-result drain. All 155,365 frames had been collected after approximately
641.6 seconds, while 45 parent postprocess jobs remained. The log shows 40
nonempty shell layers; the original serial projector repeated source geometry
calculations across 17,879,916,848 output voxels for each layer.

Radial source projection now builds exact 2D plane ownership and periodic-column
tables once and reuses them across the source stack. Compiled, nogil CPU gathers
honor the requested worker count within explicit output-buffer limits. Tables
are immutable and shared across compatible tilt/height variants; per-layer
metadata retains the precise ideal and rounded-sample shear rules. Known source
bounding boxes skip impossible mask reads. Callback delivery stays ordered,
bounded and fully settled before borrowed input or output ownership can end.
The original NumPy implementation remains the numerical reference and bounded
fallback for unavailable compilation or unusually large geometry plans.

For non-tiled, non-interpolated views whose dense canvas will immediately retire,
the original canvas remains alive until terminal-ref validation. It is no longer
copied to an additional multi-GiB native file only to delete that file. Existing
backings, retained debug artifacts and tile ownership keep their previous rules.

The projector prints flushed planning/start/completion lines including backend,
actual workers, plan size, cache hits, setup time, total time and sink wall time.
Use these to distinguish source projection from final NRRD/video output.

`tools/benchmark_radial_projection.py` exercises true 3072-square mask strides
and the production output geometry with deterministic synthetic masks. Sampled
slabs across all three bases matched the unchanged reference exactly; four-worker
comparisons were 7.4–9.6x faster without source bounds and 10.0–15.5x with bounds.
A complete 17,879,916,848-voxel compiled stream took 82.9 seconds with eight local
CPU workers and a checksum sink; sampled slices were independently compared.
These measurements exclude model inference and NRRD encoding and are not a
prediction of cluster wall time. Full cluster timing remains to be checked.

### Development v20.0.3 CUDA source projection

The completed v20.0.2 cluster job 142444 took 1,927.5 seconds, including a
1,249.0-second post-inference tail. All 40 nonempty Radial layers completed on
the compiled CPU backend. That implementation remains the CPU fallback.

Radial source projection now attempts CUDA by default. It uploads the cleaned
uint8 mask and the exact existing plane/metadata tables to an admitted device,
then applies the same periodic OR gather in bounded output blocks. Geometry and
trigonometry remain in the verified host tables; device height arithmetic uses
float64 round-to-nearest operations, ties-to-even height selection and 64-bit
source addresses. Numba is not required for this CUDA route.

The existing main-process stage coordinator chooses among the configured worker
GPUs and holds an exclusive lease through upload, projection, sink callbacks and
cleanup. Normal inference-priority and terminal asset-retirement rules apply;
the legacy idle-device backprojection exception is not used. A layer that cannot
obtain a device proceeds on CPU without blocking inference or other parent jobs.
Actual free VRAM must cover the full mask, geometry, bounded output and reserve.
Unsupported masks, absent CUDA/CuPy, insufficient VRAM or settled setup failures
select the retained CPU route before publishing any output.

Private device and pinned-memory pools avoid global allocator-cache flushes.
Upload and output staging are capped at 64 MiB each, with up to two independent
host result blocks. One GPU producer overlaps the current block's CPU mask-store
callback. Callback order and source ownership are preserved on cancellation and
failure. A late GPU error is propagated to the caller's transaction rather than
restarting into a partially populated sink; a stream that cannot be fenced is
fatal and retains its device lease and allocation owners.

`Radial projection start ... backend=cuda_factored ... device=cuda:N` identifies
successful CUDA admission. A flushed `Radial CUDA fallback` line explains a busy
device or failed admission. Set `YOLO_TTA_GPU_RADIAL_BACKPROJECT=0` to force the
v20.0.2 CPU route. `tools/benchmark_radial_projection.py` remains CPU-only;
`tools/benchmark_radial_cuda_projection.py` qualifies CUDA with production mask
strides, independently checked source slices and an optional full output stream.

Local CUDA qualification covers all three bases, tilted views, reduced processing
grids, periodic repeats, empty masks, half-even ties and neighboring float64
values. The full 17,879,916,848-voxel transverse stream matched the v20.0.2 CPU
stream's complete SHA-256 and independently checked NumPy reference slices.
It took 12.35 seconds including upload from a warm host cache and checksum work;
the earlier eight-worker CPU stream took 82.88 seconds. A separate initial source
upload took 10.71 seconds on the local Oculink-connected GPU. These are synthetic
component measurements excluding inference and NRRD/CVOL encoding, not cluster
wall-time predictions.

Thirty actual native masks captured from the cropped real-volume PT pipeline
were replayed through public CUDA dispatch after inference drained. All 47,185,920
output voxels matched the unchanged NumPy pull oracle exactly. The normal small
pipeline run exercised busy-device CPU fallback while inference owned the GPU;
its 64 decoded native/LQ NRRDs retained identical payloads.

### Development v20.0.4 compact CUDA publication

Job 142462 completed 36 Radial layers on CUDA and four on the early CPU fallback,
using all four configured GPUs. Its application timer recorded 1,364.5 seconds,
including a 696.1-second post-inference tail. CUDA layers had median total/setup/
sink times of 54.38/13.11/35.94 seconds. These phases overlap across workers; the
remaining time after subtracting setup and sink is not a GPU kernel timer.

For an incremental CVOL sink, CUDA now derives tight slice bounds and foreground
counts from its completed mask blocks, then concatenates normalized crops or
little-endian row-packed bytes on device. It transfers only those bytes and small
metadata records. The writer validates slice order, bounds, counts, payload layout
and packed-row padding before reserving one append for the block. It updates the
same CVOL index/extents without a CPU bounding-box scan, normalization, foreground
recount or packing pass. Public NRRD layers retain raw CVOL payloads; packed output
remains an internal retention format. Generic callbacks and dense return values
continue to receive dense blocks.

Two device output buffers are capped at 64 MiB each, alongside bounded metadata
and independent readonly host payloads. Compact output is preflighted before
publication; the existing admission, failure transaction and device-fence rules
remain in force. Completion logs identify `backend=cuda_factored_compact`, the
payload format, device-event timings and actual metadata/payload/dense D2H byte
counts. These distinguish GPU computation, transfer and CPU publication waits.

Concurrent requests for identical plane geometry now share one build Future.
Unrelated keys still build concurrently. Cache clearing detaches older requests
without cancelling their waiters or allowing old builds to repopulate a new cache
generation. The 256 MiB readonly LRU and numerical plane construction are unchanged.
`plan_s` includes waiting on a shared build, so a slow reported hit need not imply
repeated computation. On Windows, the incremental writer uses serialized
seek/write/restore when `os.pwrite` is absent; POSIX positional writes are unchanged.

Qualification through the actual CVOL writer decoded the entire 17,879,916,848-voxel
synthetic fixture to the same SHA-256 for dense/raw-compact/packed-compact output
and the previous CPU stream. D2H fell from 17,879,916,848 bytes to 246,161,064 bytes
for raw crops and 30,847,944 bytes for internal packed crops, including metadata.
The fixture has only 48 nonempty source slices; production masks are less sparse.
Dense and compact benchmark source uploads had different cache warmth, so their
total-time ratio is not a controlled speedup claim. Cluster wall time remains to
be measured. `tools/benchmark_radial_cuda_sink.py` records upload, device, sink and
decode timings separately.

Twenty-nine actual PT masks were replayed through compact CUDA dispatch and the
real CVOL writer after inference drained. All 45,613,056 decoded voxels matched
the unchanged NumPy oracle. All 64 real-run NRRD payloads and both videos' decoded
frames/timestamps matched the v20.0.2 baseline.

### Radial setup optimization after job 142502

Job 142502 took 1,311.9 seconds, including a 670.9-second post-inference tail.
Of 40 nonempty Radial projections, 36 used compact CUDA and four took the early
busy-device CPU fallback. CUDA median total/setup/plan times were
45.82/26.38/11.96 seconds, versus 0.69 seconds of projection kernels and 1.90
seconds of sink callbacks. Shared-plan waits and concurrent layers overlap;
these measurements cannot be summed into pipeline walltime.

Plane construction now evaluates the original NumPy equations once per bounded
strip and assembles the same readonly CSR tables. One-million-pixel strips reduce
interpreter handoffs during concurrent output work. The two-pass constructor is
retained as a qualification reference. Final plan admission and the 256 MiB cache
are unchanged. Temporary occurrence arrays are bounded by the admitted column
budget; final concatenation briefly retains two copies of the column payload.

When valid source bounding boxes reduce storage, CUDA packs their rectangles
directly from strided source masks into bounded pinned upload staging. It admits
and uploads the concatenated uint8 payload instead of a complete dense mask.
The existing bbox guard and geometry tables remain authoritative; a uint64
per-shell offset only changes storage addressing. Empty shells need no payload.
Without useful bounds, the dense upload path remains available. No additional
foreground scan, full host packing array, or model-output change is introduced.
Stage leases, preflight, ordered publication and fatal fencing rules are retained.

Start/completion logs include `source_layout`, logical `source_bytes`, actual
`source_h2d_bytes`, `source_upload_seconds`, `geometry_upload_seconds`, and
`preflight_seconds`. This exposes source setup separately from projection and
compact output transfer.

`tools/benchmark_radial_setup.py --output-dir <scratch-directory> --gpu --full-stream`
checks all ten production plane plans against the original construction, heats
the GPU for 60 seconds, then compares dense/cropped/cropped/dense input uploads
with identical production-stride synthetic masks and warmed host pages. Local
plans matched exactly and built about 1.6–2.0x faster. The full tilted-transverse
17,879,916,848-voxel output stream had identical SHA-256 in all four runs, with
independent NumPy checks on three source slices. Source H2D fell from
11,966,349,312 to 213,516,288 bytes; upload took 2.78–2.84 versus 0.163–0.165
seconds on the RTX 4090 Laptop GPU. These masks are synthetic rectangles; this
does not establish production sparsity, output contention or cluster walltime.
Raw and packed CVOL integration additionally exercises the public dispatcher,
actual CUDA crop upload, encoded callbacks and full decoded-store equality.

### Radial host setup after job 142515

Job 142515 did not validate an end-to-end improvement: walltime rose from
1,311.9 to 1,390.8 seconds (+6.0%), split into +25.2 seconds through inference
drain and +53.7 seconds in the tail. All 36 CUDA layers used cropped sources,
reducing their logical 324,988,305,408 source bytes to 53,936,219,196 H2D bytes.
Maximum plan wait fell from 85.29 to 19.95 seconds, but median total CUDA layer
time increased from 45.82 to 56.41 seconds. Setup still had a median 14.52
seconds outside its reported plan/upload/preflight subphases. These concurrent
layer intervals do not sum to critical-path walltime or identify its sole cause.

Tilt metadata now batches the original NumPy sampled-shear operations across
shells, with at most one million intermediate values per batch (or one native
row for unusually wide input). Source crop uploads use optional cached Numba
packing into the existing bounded pinned buffer. Validated uint64 linear byte
indices allow contiguous vector copies without Python handoffs between shells.
Unavailable compilation retains the NumPy uploader before any source upload;
later runtime failures still fence and fail the constructor transaction.

Logs partition host metadata, CUDA admission, contract validation, module setup,
buffer allocation, source packing and constructor time. Admission includes the
constructor, packing is part of upload, and other subphases are nested rather
than additive independent walltimes. CUDA event intervals around separately
enqueued operations may also include host enqueue gaps; they should not be
interpreted as isolated kernel execution times under host contention.

`tools/benchmark_radial_host_contention.py --output-dir <scratch-directory>` uses
real XTA sparse-store NRRD writers concurrently with production-stride synthetic
Radial setup. In alternating original/candidate/candidate/original trials with
eight writers, the qualified vector-copy candidate reduced median host metadata
from 0.3065 to 0.1213 seconds and source upload from 0.8624 to 0.6103 seconds.
Combined metadata and CUDA construction fell from 1.2888 to 0.8565 seconds.
All metadata bits and three full GPU output slices matched; 291 background NRRDs
completed and the last file from each of eight writers decoded exactly. An
initial Numba slice-copy variant regressed and was replaced before qualification.
The local benchmark excludes inference, multiple GPUs and low-quality mirrors;
the new candidate still requires a cluster walltime measurement.

### Job 142523 and disabled dispatch experiments

Job 142523 took 1,375.3 seconds, versus 1,390.8 for 142515. Time through inference
drain fell by 22.6 seconds, while the tail grew by 7.1 seconds to 731.7 seconds.
All 36 CUDA layers selected the nogil crop packer. Median setup/upload times
fell from 24.88/4.37 to 7.08/1.62 seconds; median time after setup increased from
26.14 to 44.22 seconds. The four early CPU fallbacks moved from coronal to
transverse views. Comparing the 32 views that used CUDA in both runs still shows
setup improving from 23.30 to 6.47 seconds and time after setup growing from
26.04 to 42.04 seconds. These overlapping phase intervals do not establish the
cause of overall walltime or isolate host, GPU and output contention.

The CUDA projector now exposes two **disabled-by-default experiments** through
its internal constructor: `use_graphs` and `skip_empty_blocks`. The pipeline
does not opt into either. Graph capture combines projection, crop metadata and
its pinned D2H copy, with an owned first-Z scalar and at most three captured
block sizes. Capture uses thread-local mode; settled unavailability retains
direct CUDA, while failed fences and late replay errors retain the existing
fatal/transactional rules. Empty-block elision can skip offset upload and the
second encoder fence after validated metadata proves a block empty. Preflight
still exercises both encoders. Graph work is timed as a combined interval in
`cuda_graph_seconds`, not misreported as individual kernel timings.

`tools/benchmark_radial_graph_dispatch.py` compares full ordered CUDA/CVOL
streams while eight real full-width NRRD writers and 0.20 mirrors run. Their
64-slice depth is deliberately bounded. Graph-only trials were mixed: median
projection/publication changed from 22.72 to 24.67 seconds for broad rectangles,
and 4.16 to 3.75 seconds for an ellipsoid with a production-like 1.17 GB output
payload. Adding empty-block elision changed 4.85 to 5.16 seconds in a repeat.
All four full decoded 17,879,916,848-voxel streams matched in each experiment.
These results do not justify enabling either experiment in production.

Jobs 142535 (reported capped) and 142543 (reported uncapped) took 1,352.6 and
1,329.8 seconds, with tails of 700.1 and 697.2 seconds. Both logs report eight
global NRRD bands, whereas this writer would allow four with the requested sink
cap. Effective settings/deployed writer provenance are therefore unconfirmed;
the pair does not conclusively isolate output contention. Defaults remain unchanged.

### Native shell-owner execution

At jobs 142535/142543, the GPU-only fast-bundle setup disabled shared direct union
globally, then excluded shells from D1. Their worker result mode was
`file`. Task masks are copied to the parent, accumulated into complete native
views, cleaned and projected under parent postprocessing. At inference drain in
both runs, the 30 older-family layers were published while 45 parent jobs
remained; only four Radial plans had started, on CPU fallback. Projected refs are
already authoritative for final fusion, so there is no additional duplicate
terminal radial projection to remove in this command.

`cylindrical_owner` now consumes each complete native radius chunk in its
inference worker and gathers directly into a persistent source-space bitset.
The existing plane plan is grouped by nearest owning shell. Each chunk visits
only its owned base-plane positions while retaining all periodic occurrences,
the ideal-height validity test, float64 sampled shear, clamp and ties-to-even
rounding. The CUDA gather sets uint32 source bits atomically. Its geometry is
the established pull relation; the legacy D1 splat kernel remains guarded.

`tools/probe_radial_streaming_architecture.py` is a small CPU correctness probe,
not a runtime backend or benchmark. Its 1,560 cases matched the existing pull
oracle across axes, tilts, wraps and processing/restoration grids, including
shuffled radius arrival and two-owner bitset reduction. The implemented first
path uses one owner per view even when multi-owner groups are requested. A
shape-only prediction target forbids host mask access and checks actual result
coverage; the owner separately checks chunk/radius coverage before publication.

Native cleanup uses four-connected background union-find within foreground
bounds, then fills only components that do not reach the crop boundary. A private
stream and reusable label buffers avoid per-radius device-wide fences and global
allocator-cache flushes. Cleanup always follows the complete radius union. Both
cleanup and projection are preflighted before the worker loads its model.

The current command selects `projection_contract=radial_native_pull_v1` inside
the established D1 owner envelope. Unsupported configurations retain their prior
route. `YOLO_TTA_RADIAL_OWNER=0` is an explicit compatibility opt-out. The owner
admits its bitset, geometry, label buffers and reserve against actual free VRAM;
resource or device failures remain loud and cannot publish a partial result.
Worker retirement refuses active native owners, and failed CUDA fences retain
borrowed/device owners through the fatal path.

Completed bitsets use the existing bounded asynchronous CPU CVOL publisher with
native-shell provenance. Native-empty views retain internal empty contributions
without adding public NRRD files. Every requested nonempty layer and mirror stays
independent. Source-bitset D2H and CPU unpack/publication remain measured costs.
The launcher logs effective and constructed output worker settings plus loaded
module paths/hashes at task planning and inference drain.

Qualification with a local one-channel PT model and all 50 shell trajectories
matched all 52 native/LQ NRRDs and both videos' decoded frames/timestamps against
compatibility execution. Multi-chunk leases were forced; the owner run recorded
50 completed owners, no host native-view unions and no parent Radial projection.
The helper excludes older physical families because their D1 path requires a
TensorRT runtime, which was unavailable locally; production view selection is
unchanged. Direct CUDA tests cover all axes, tilts, reduced/restored grids,
3072-square cleanup, coverage/failure rules and real CVOL publication.

The production-grid synthetic ellipsoid produced the same complete decoded
17,879,916,848-voxel SHA-256 as the retained projector. After a 60-second GPU
heatsoak, native cleanup/gather took 3.17/0.21 seconds, bitset D2H 0.68 seconds,
and CPU publication 7.27 seconds on the local GPU. Fixture uploads and final
validation decoding are separate; these are component measurements.

Cluster job 142565 subsequently completed the full four-GPU TensorRT command
in 1,124.5 seconds versus 1,329.8 seconds in job 142543, a 205.3-second (15.4%)
reduction. Its post-inference tail fell from 697.2 to 434.9 seconds. All four
workers passed native Radial preflight, all 50 shell views (40,915 frames)
used the native owner contract, and all 155,365 model frames completed with
zero parent postprocess jobs at inference drain. The time through the drain
increased from 632.6 to 689.6 seconds as shell cleanup/projection moved into
the workers. These are individual end-to-end runs; exact output parity was
qualified separately with the local decoded-output comparisons above.

### Main v20.0.3 output tail and memory planning

Owner publication now scans source words directly for exact slice bounds and
foreground counts, then writes cropped, row-packed bits. It avoids constructing
the dense uint8 publication blocks. The existing packed CVOL reader supplies
final union and all requested layer products. Optional compilation failure uses
bounded NumPy unpack/pack; `YOLO_TTA_PACKED_OWNER_PUBLICATION=0` restores raw CVOL
publication. Packed Windows descriptors explicitly use binary mode so byte 0x0a
cannot be translated into CRLF.

For native CVOL NRRDs using software member codecs, the writer emits only the
nonempty crop's row bands to the regular codec. Empty top/bottom rows and slices
use reusable gzip zero members, restricted to 21 power-of-two sizes through
1 MiB. These zero members use compression level 9 once per cached size; ordinary
data retains the configured codec. The completion queue remains bounded, and
every sparse mirror observer receives its complete crop once. Restored geometry,
dense-block observers and hardware minimum-input policies retain their existing
paths. `YOLO_TTA_NRRD_CROP_ROW_SPANS=0` restores whole-slice assembly.

Linux native shell payloads can use parent-owned memfds under a run-wide plan.
Every selected future layer is charged its full worst-case packed source size
before dispatch, so workers cannot independently claim the same free RAM. The
plan counts physical/cgroup headroom without swap, reserves the final union and
uint32 topology labels, all host publication credits, configured gzip windows,
mirror canvases and global spools, and leaves half the remaining headroom unused.
The standard geometry with 950 GiB headroom reserves 291.42 GiB for other work
and admits 50 layers under a 104.14 GiB worst-case retained-payload reservation.
Actual sparse payloads are much smaller. Parent descriptors survive worker exit;
existing consumer retirement closes them. Between producer callbacks, inadequate
headroom or grant exhaustion spills the unfinished payload to disk and releases
its RAM pages. Disk-write failure preserves the original payload and propagates.
Retained-debug runs and memory-backed scratch use their existing backing policy.
`YOLO_TTA_PUBLICATION_RAM=0` disables this tier; a positive
`YOLO_TTA_PUBLICATION_RAM_GIB` adds a retained-payload cap. Existing anonymous
workspace caps are also respected.

Topology adjacency can intersect equal-label row runs rather than inspect every
foreground pixel's neighboring labels. Exact sorted pair codes retain the same
union-find result. Small overlap windows use the existing pixel hash, while
fragmented inputs exceeding bounded run/pair buffers fall back to it.
`YOLO_TTA_TOPOLOGY_RUN_ADJACENCY=0` disables the run path. The 3072-square coherent
fixture was about five times faster for adjacency; this is not a measurement of
the full production keep_objects pass.

The full 17,879,916,848-voxel synthetic publication comparison measured 17.5–18.1 s
for raw publication plus native/mirror NRRDs, versus 10.1–10.6 s for packed
publication and crop-row streams. Temporary payloads fell from 3.223 to 0.404 GiB;
native NRRDs fell from 91,726,928 to 53,183,381 bytes. Complete decoded native and
mirror hashes matched. All 52 NRRDs and both videos from the 50-trajectory local
real-model qualification matched the prior implementation, including decoded
video frames and timestamps. These results exclude cluster scheduling and RAM
backing speedups. The Linux memfd spawn/retirement test is included but needs a
Linux host; the Windows run qualifies admission arithmetic, real spill copies,
failed-write recovery, packing, and the GPU worker/consumer pipeline.

### Cluster 142619: RAM payload cache identity

Job 142619 passed the original single-store Linux memfd ownership test and
admitted all 50 native layers to planned RAM. It later failed while reading a
Radial payload for NRRD export. The shared-mmap cache used `Path.resolve()` as
its key: distinct, equally named memfds resolve through `/proc/<pid>/fd/N` to
the same diagnostic `/memfd:xta-packed-publication (deleted)` string. A reader
could therefore receive another layer's mapping. Different payload lengths
produced a short read; equal-length payloads could produce incorrect pixels
without a read error. The per-view exports from that failed run are untrusted.

The cache now uses the absolute logical layer path without following its payload
symlink. Repeated readers of one layer still share a mapping, separate layers
remain separate, and release/invalidation keep the same key after descriptor
retirement. Cross-platform regressions reproduce the old collision, including
equal-length/different-content layers, and compare concurrent native/mirror
exports against uncached output. The existing `tests.test_publication_memory`
preflight now also checks several real Linux memfds after their producer exits,
through concurrent cached NRRD export. This extends the original one-store test.

The inference/projection and optimization policies are unchanged by this fix.
In 142619, keep_objects took 27.057 s versus 81.903 s in 142565; both report
408,973,828 retained voxels. This count is not a full-volume parity proof, and
142619 did not complete successfully or report a completed pipeline walltime.

### Cluster 142621: completed tail qualification

The corrected candidate completed successfully in 620.5 s, versus 1,124.5 s in
142565: 504.0 s (44.8%) less walltime. The post-inference tail fell from 434.9 to
83.5 s (80.8%), and keep_objects fell from 81.903 to 21.535 s. Time through the
drain fell from 689.6 to 537.0 s.

All 12 preflight tests passed on Linux, including concurrent cached exports from
multiple parent-owned memfds after producer exit. All 155,365 model frames and
71 NRRD write jobs completed, no traceback was logged, and the run released all
50 RAM-backed payloads before reporting Done. The production payloads did not
spill to disk. The retained voxel count remains 408,973,828; this count and the
successful writes establish logged completion, not a separate full-volume byte
comparison. These changes are promoted in main v20.0.3.

## Default sparse execution

Full-frame, zero-angle Cartesian interpolation components retain sparse storage
through publication. Transverse components reuse their immutable store;
sagittal and coronal components transpose directly into packed orthogonal stores.
Azimuthal and tilted-Azimuthal components invert the reference projector's discrete
ownership map and visit foreground crops directly, producing packed source-space
stores without dense view reconstruction. Compiled kernels use the existing
optional Numba dependency; ordinary tilted Cartesian components retain the
dense projection backend.

Eligible D1 continuations publish complete component references across view
families and angles without constructing a dense additions volume. Their
independently published source-space base remains part of the final union.
Tiles, retained debug workspaces, no-NRRD runs, and configurations without complete
component coverage retain the ordinary continuation path. Membership export uses
validated paste bounds and foreground counts to encode cropped slices, with a
full-slice fallback when the bounds cannot establish the same component payload.

Immutable components pass to a separate projection executor, freeing parent
preparation slots before publication completes. Admission bounds pending input
bytes and estimated active scratch independently; one oversized job may run alone.
The parent remains a scheduler dependency until every component future has settled,
and failures wake blocked producers before executor teardown. The handoff preserves
layer identities and every requested NRRD and downsampled overlay.

After inference results and D1 ownership drain, each CUDA worker fences and releases
its rendering source, texture, model, graph, and allocator assets. An acknowledged
barrier precedes post-inference GPU admission. A validation refusal retains the
worker's assets and existing memory admission; a failure after release begins is
fatal. Successful release permanently closes that worker to further inference
while leaving it available for auxiliary mask work.

Sparse interpolation labels use validated foreground bounds to select cropped
CPU labeling at low coverage. Touching component pairs use bounded compiled
deduplication when Numba is available, with the NumPy implementation retained
for unsupported inputs or unavailable acceleration. Final fusion directly ORs
sparse crops when only temporal restoration is needed, preserving the existing
source-index mapping. These paths need no experiment-specific environment flags.

The GPU external augmentation examples use separable Gaussian operations.
GPU bridge painting, interpolation radius experiments and existing hardware
feature controls retain their separate policies. No additional dependencies are
required beyond the existing optional acceleration dependencies.

Linux scratch classification uses the kernel mount ID of an opened path, so a
job-private bind mount takes precedence over covered host mounts. Missing paths
are classified through their nearest existing ancestor. Filesystem type does
not imply persistence: cluster `/tmp` remains disposable after the job ends.

## Runtime modules

| Module | Ownership |
|---|---|
| `config` | CLI grammar, validated selections, channel/view request formats, version constants |
| `workspace` | scratch policy and environment-derived workspace settings |
| `runtime` | telemetry, NUMA primitives, memfd/workspace ownership, executors and process setup |
| `media` | ffprobe/ffmpeg, decode/resize readiness and source-volume lifecycle |
| `render_batch` | runtime frame-carrying `RenderBatch` values and exact model/image fan-out contracts; distinct from logical `RenderRequestBatch` planning |
| `geometry` | authoritative CPU forward renderer, affine/view/channel/seam/tile primitives, `RasterPlan` builders, slicing and render-source geometry |
| `cylindrical_geometry` | dense annular shell grids, periodic intrinsic patches, zero-extended source sampling and radius trajectory coordinates |
| `cylindrical_projection` | bounded source-coordinate pull projection, discrete tilted-sample ownership and periodic occurrence union |
| `gaussian` | one binary Gaussian numerical primitive shared by PTA preprocessing and TTA postprocessing |
| `inference` | shared Ultralytics execution, mask payloads and inference cleanup |
| `cuda_backend` | CUDA-resident rendering and CUDA worker-side helpers |
| `cuda_interpolation` | Lazy CUDA bridge morphology/radius evaluation and crop-bounded painting |
| `workers` | module-level OpenVINO and CUDA worker entry points |
| `topology` | slice labeling, union-find and component metadata |
| `backprojection` | azimuthal/tilted projection plans and source-space accumulation |
| `sparse_projection` | exact crop-driven Azimuthal/tilted-Azimuthal inverse ownership maps and packed source-space publication |
| `projection_queue` | bounded immutable-component handoff, active scratch admission and projection-future lifetime |
| `component_replay` | bounded persistent component capture, checksummed geometry descriptors and replay loading |
| `finalization` | source-volume fusion, object filtering and centerline processing |
| `interpolation` | interpolation planning, execution and sparse continuation |
| `cuda_d1` | D1 owner-GPU backprojection and packed source-space storage |
| `experimental_features` | dependency-light, version-neutral feature gates for opt-in hardware experiments |
| `cuda_finalization` | dependency-light distributed-binary contracts plus the opt-in transactional multi-GPU keep tail |
| `assembly` | completed-view preparation, tile gates and smoothing handoff |
| `outputs` | NRRD, TIFF/MKV, summaries and low-quality derivatives |
| `unification.contracts` | dependency-light `ForwardSamplingPolicy`, digest-addressed `RasterPlan`, logical `RenderItem`/`RenderRequestBatch`, channel/tile and data-role contracts |
| `unification.sampling` | the executable forward-policy registry, fail-closed backend/role binding, execution records and canonical plan factory |
| `unification.channels`, `unification.tiles` | shared single-channel-layout expansion and strict grouped tile parsing |
| `unification.runtime`, `unification.views` | the shared TTA-authoritative physical-view compiler and dependency-light grouped view requests |
| `unification.context`, `unification.manifest`, `unification.tta_manifest` | launch context, atomic JSON publication, artifact identities and mode-qualified v18 manifests |
| `pta_config` | strict PTA-only grammar, grouped geometry and mode-specific defaults |
| `pta_mode`, `pta_runtime` | dependency-light PTA validation followed by construction of the complete native runtime option contract |
| `pta_scheduler` | independently tested CUDA-owner layout, compatible-work packing, VRAM admission and deterministic OOM splitting |
| `pta_augmentation` | external PTA policy inspection, loading, deterministic thread-local construction and paired image/mask execution |
| `pta_dataset` | PTA candidate identity, deterministic augmentation-version planning, background policy and dataset splitting |
| `pta_rendering` | canonical spawn-pickled PTA render-plan values, CPU geometry caches and frame/tile rendering primitives |
| `pta_publication` | image/label encoding, atomic publication, canonical dataset image sink and candidate output paths |
| `pta_workers` | sole PTA shared-memory, worker-global, CPU/GPU task-entry, result and persistent-pool process owner |
| `pta` | PTA source discovery/preprocessing, geometry planning, dataset orchestration, reporting and manifest publication |
| `lta_config`, `lta_mode` | dependency-light LTA grammar and deferred runtime dispatch |
| `lta_inputs` | class-0 YOLO-seg target/exemplar discovery, full/partial annotation states and deterministic prompt ranking |
| `lta_sam` | local-only SAM bundle boundary, exact installed SAM source-tree provenance, fixed 30-frame leases and runtime-neutral image/video result contracts |
| `lta_runtime` | fail-closed LTA preflight plus physical-view/angle/tile/session/device planning and deferred production execution dispatch |
| `lta_outputs` | role-aware LTA layer recomposition, unconditional terminal-union primitive and atomic complete-manifest writer |
| `lta_postprocessing` | non-mutating prediction/dogfood hole filling, completed-view cleanup, scalable native-union postprocessing ownership and atomic final-union NRRD publication |
| `lta_propagation`, `lta_windows` | stable authoritative-mask session contracts, hole-filled temporal dogfood and fixed 30-frame center-out chains |
| `lta_workers`, `lta_worker_adapter` | spawn-isolated one-model-per-GPU execution, compact artifact transport and retry-safe worker lifecycle |
| `lta_execution` | native Transverse production coordinator, monotone cross-tile mask fixed point, storage/watchdog admission, compact worker audit, one-time backprojection and complete publication transaction |
| `lta_rendering` | XTA-authoritative full-frame LTA raster rendering, implicit RGB conversion, inverse in-plane mask restore and source-space backprojection seam |
| `lta_tiles`, `lta_tile_tracking` | concrete edge-pinned grids, eight-neighbor overlap topology, bidirectional spatial relay and global-coordinate overlap evidence |
| `lta_tracklets` | anchor-scoped matching, residual split/merge hypotheses and authoritative temporal handoff |
| `lta_experimental` | narrow pinned adapter for authoritative-mask injection and direct tracker-only propagation |
| `lta_scheduler` | physical-view GPU affinity, atomic session ownership, tail assistance and exactly-once backprojection admission |
| `tta_mode` | production TTA runner entered only after mode-specific CLI validation |
| `tta_lifecycle` | outer TTA resource ownership, selected-run scratch cleanup and complete-manifest publication transaction |
| `tta_outputs` | single-use identity-preserving ownership and ordered teardown of settled TTA output artifacts |
| `tta_prediction` | lazy physical-view frame caching and bounded prediction-source build, staging and warmup queues |
| `tta_scheduler` | stateful TTA process-inference admission, hybrid/D1 ownership, CPU/GPU dispatch, result transport and accounting |
| `tta_terminal` | completed physical-view TTA collapse, one-time terminal backprojection and dense-union handoff credit |
| `cli` | dependency-light strict `--mode tta|pta|lta` dispatcher |
| `intel_compression` | dependency-light policy/adapter for optional QATzip and QPL companion extensions |
| `intel_dsa` | lazy policy, eligibility, drain transaction and lifecycle for optional Linux idxd workspace copy |
| `pipeline` | production TTA preparation, completed-view assembly and output facade around the process scheduler |

### Incremental orchestration decomposition

The current decomposition establishes explicit owners for TTA run resources/publication, prediction-source
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

### Intra-node HGX experiments

Two experimental paths are dark by default and retain the established reference behavior:

- `YOLO_TTA_D1_OWNER_GROUPS=1` allows a D1 parent whose complete seed leases
  need no view-shadow writer to bind deterministic slice coverage to several idle CUDA
  workers. Admission is a centralized, atomic pre-dispatch reservation rather than a side
  effect of pending-task feasibility scans. The scheduler keeps one group active at a time, chooses
  the heaviest still-unclaimed eligible parent, and plans the next group after the prior
  reduction releases its participants; nonparticipants retain ordinary view-level parallelism.
  Each participant retains a dedicated IPC-exportable partial bitset until the scheduler
  acknowledges either CUDA-IPC/NVLink reduction or bounded host-path recovery, then confirms
  an explicit release acknowledgement from every participant before reusing the workers.
  `YOLO_TTA_D1_OWNER_GROUP_SIZE` caps the participant count at the job-visible
  device count. The one-owner path remains the admission and execution fallback.
- `YOLO_TTA_GPU_RESIDENT_TAIL=1` tries the first Track-A transaction after the
  inference workers have drained: the settled host final union is uploaded into contiguous
  job-visible Z shards, exact 26-connected labels stay device-resident in memory-bounded
  3-D CCL blocks, and compact equivalence pairs are needed only at block/shard boundaries.
  Area and boundary metadata is resolved through the CPU union-find reference, and a separate filtered candidate is
  committed only after every GPU succeeds. `YOLO_TTA_GPU_RESIDENT_TAIL_REQUIRED=1`
  makes failure fatal for qualification; otherwise the untouched host union enters the
  established CPU `keep_objects` path.

`cuda_finalization.DistributedBinaryArtifact` is the common future handoff boundary for
host uint8 volumes, D1 bitsets, resident final-union shards, and a later multi-GPU
interpolation producer. The Track-A implementation exercises the host-upload
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

### LTA stabilization contracts

LTA plans concrete edge-pinned tile grids and computes each tile's actual Moore-neighborhood
overlap. Every contiguous qualifying overlap episode emits a forward relay from its first shared
frame and a backward relay from its last shared frame. Those two sessions cover both
destination-only tails and the crossing interval itself; disjoint leave/re-entry episodes remain
distinct. Relay masks are rebased in global view coordinates, carry globally scoped lineage
identity, and merge when several neighbors reach the same destination event. The idempotency
identity includes lineage, destination, frame, and temporal direction: an exact ping-pong event is
suppressed, while complementary foreground from a longer route advances an accumulated mask
revision and an object that leaves and later re-enters a previously visited tile is admitted. Tile
ancestry remains audit evidence rather than a permanent exclusion rule. A polygon is injected in
one strongest tile whenever that tile contains its complete mask; polygons larger than any tile
retain every required authoritative fragment.

The device plan assigns each `(volume, physical view)` to one GPU owner. That owner publishes one
immutable rendered-view cache and remains the sole backprojection owner across all angles, tiles,
anchors, and sessions. After a GPU drains its own views, it may steal an unopened atomic session
from the heaviest remaining owner queue and consume that existing cache. A live SAM session never
migrates between devices, and completed work commits in plan order regardless of finish order.
The public runtime now executes the native Transverse, angle-zero, overlapping 1008-pixel tile
contract. One persistent spawned worker owns each selected GPU. Seed groups are deterministically
batched at SAM's 128-object limit; filled predictions stream directly into file-backed union and
bit-packed relay reducers; only boundary dogfood masks and compact audit state remain resident.
The tracker confidence is applied to the sigmoid framewise score, while exact removal sentinels
remain bookkeeping rather than zero-confidence predictions. Settled generations feed a bounded
breadth-first eight-neighbor relay fixed point; a safety-cap hit with remaining mask growth fails
instead of publishing a truncated result. Each worker verifies the exact pinned SAM distribution,
commit/tree, and BPE before model construction. Compact per-device runtime/profile, relay-gate,
confidence, hole-fill, dogfood, and termination evidence survives temporary artifact cleanup.
Results collapse in physical-view space, receive a
final 2-D hole fill, backproject exactly once, apply the structured TTA postprocessing order to
the native union, restore immutable hard-positive foreground after destructive filters, and
publish `Global_final_output` plus the complete manifest.
Non-Transverse or nonzero-angle execution remains fail-closed until provisional-volume bootstrap
is separately qualified; it never falls back to the inferior box/composite experiments.

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
  match the batch plan digest. Synthetic Cartesian tail repeats and azimuthal seam-extension slots
  remain explicitly marked so artifact sinks can omit them.

Only layout, dtype, normalization, and other backend-only conversion may occur after the
frame-carrying boundary. Geometry may not be independently reconstructed for an image sink.
CPU-backed main-process and OpenVINO/CUDA-worker slab sources implement this fan-out today;
device-resident CUDA/direct TensorRT-ring capture remains unfinished and unqualified.

Unification also uses shared operation primitives rather than duplicating mode adapters.
`unification.channels` expands the canonical TTA channel grammar into TTA ascending or PTA
ascending/reversed variants; shared geometry owns contextual addressing and azimuthal mirror parity.
`unification.tiles` parses each strict `TILE_SIZE:TILE_STRIDE` group, while `geometry` builds the
collapsed direct-to-output tile transform. `gaussian.binary_gaussian_pass` supplies the one
constant-zero, truncate-4, threshold-at-0.5 numerical operation; mode-owned orchestration decides
whether it runs before geometry (PTA) or after fused prediction (TTA).

Grouped-view duplicate behavior deliberately follows TTA. Exact repeated tilted groups and
overlapping tilted groups that generate the same concrete signed-angle/direction view are
deduplicated. Duplicate tile groups are errors. Duplicate Cartesian tokens and repeated
assignment of a azimuthal target remain errors because those forms are ambiguous under their
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

PTA accepts only the resolved mode configuration. Source discovery, geometry planning,
publication and validation run through that contract; there is no separate legacy
single-volume renderer or completion-marker resume path. The self-contained GPU example
policies use separable one-dimensional Gaussian passes for blur and elastic-field smoothing.
Their affine inversion and policy seed/selection rules remain independent of that filter.

## Output ownership and successful-run publication

Ephemeral mapped scratch uses shared mappings and does not request synchronous writeback;
same-host readers observe dirty pages through the mapping. Raw cvol payloads use ordinary
pathname-backed files. This is separate from the active memfd workspace allocator used for
shared source/result buffers, which retains explicit ownership and release accounting.

Cluster outputs and diagnostic logs must use persistent storage when they are needed after
job completion. A pathname-backed allocation on `/tmp` may be tmpfs or local SSD, depending
on the allocation; neither implies post-job persistence.

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

The v20 shell tests include actual source-tap coverage, the v19 Azimuthal pixel
reference captured before the rename, periodic duplicates, radius-only context,
thin tilted boundaries, and intrinsic-patch/Tile separation. On a live CUDA host:

```powershell
$env:XTA_RUN_CYLINDRICAL_CUDA_SMOKE = '1'
$env:CUPY_CACHE_DIR = Join-Path (Get-Location) 'build/cupy-cache'
python -B -m unittest -v tests.test_cylindrical_cuda
```

Local v20 validation on 2026-09-07 passed 819 tests with 74 skipped in aggregate
discovery, plus the explicitly enabled CUDA corpus. Independent rendered-mask
projection checks covered 312 intensity and 312 categorical cases. A bounded
24-frame real-volume crop and the supplied gray PyTorch model completed both
922-frame Radial inference and a 3,630-frame mixed Azimuthal/Radial inference run
with 0/45-degree rotations, interpolation and nested Tiles. Native/LQ NRRDs were
decoded and both output videos verified. This establishes functional integration,
not full-volume accuracy or H100/TensorRT production qualification.

Run the interpolation numerical corpus separately with the numerical dependencies installed;
the aggregate dependency-light suite can replace those dependencies with stubs:

```powershell
python -m unittest discover -s tests -p test_interpolation_geometry.py -v
python -m unittest discover -s tests -p test_external_augmentation_examples.py -v
```

Hardware-backed CUDA, TensorRT, OpenVINO, QAT, IAA, DSA and full data/model parity tests
still require the production environment and representative artifacts. Intel accelerator
build, provisioning, and admission instructions live in ``native/README.md``; run
``python tools/intel_accelerator_selftest.py --backend all`` on the target host.

The production four-GPU LTA interface is launched through the compatibility launcher. Its
one-versus-four-device scheduling and final bytes are covered by deterministic software tests;
representative H100 execution remains a hardware qualification step:

```bash
python -u GPT-6-Astra-Ultra_v21.0.2_SLURM.py \
  --mode lta \
  --input <target-video> \
  --exemplar <aligned-image-yolo-directory> \
  --exemplar_index_origin 1 \
  --output <new-output-directory> \
  --temp "$SLURM_TMPDIR" \
  --model <local-sam3.1-bundle> \
  --device 0 1 2 3 \
  --enable_cartesian transverse \
  --enable_tile 1008:756 \
  --angle 0 \
  --sam_execution video \
  --postprocessing 3d_void_fill gaussian_smoothing:3:1 \
  --save overlay voxel_volume summary
```

The aligned exemplar indexes address decoded target frames; they are not a cross-volume visual
transfer request. `<input-stem>_Global_final_output.seg.nrrd` and `manifest.json` are
unconditional. Omitting
`--tile-index` is implicit in the production CLI: the complete overlapping grid is scheduled,
and spatial relays may seed initially unannotated neighbors. Non-Transverse views and nonzero
angles remain fail-closed pending provisional-volume bootstrap qualification.
The representative 1929×3064×3024 volume with all three terminal filters needs roughly
180–200 GiB of fast scratch; execution performs a filesystem-capacity preflight before decode,
but relay growth and public output staging remain additional workload-dependent costs.
Append `keep_objects:N` only when `N` is the intended number of connected 3-D objects; `N=1`
deliberately removes every predicted component except the largest before hard-positive restoration.
The representative aligned export resolves 237 polygon lineages into 281 direct tile seeds:
231 lineages have one complete-mask owner, while six oversized polygons retain 4–13 clipped
authoritative fragments. Explicit empty labels are audited against the final union as known
background; they are not silently subtracted from model output.

The bounded LTA hardware seam can be exercised from a source checkout with
``python tools/lta_gpu_smoke.py --model <local-sam3.1-bundle> --input-root <input> --exemplar-root <exemplars> --case direct --start-frame <first> --prompt-frame <anchor> --exemplar-index <index> --label-row <row>``.
It validates one fixed 30-frame visual-prompt/video-tracking session without publishing
images or labels. The tool requires an existing local checkpoint and pinned local BPE asset;
it never downloads model material. ``--case composite`` separately exercises cross-image composite
conditioning, while ``--case both`` reuses one predictor for the two bounded sessions. The
pinned SAM 3.1 adapter suppresses the upstream tracker-only load of the merged checkpoint,
assigns one memory-mapped assembled state into a CPU parameter shell (or an explicit diagnostic
meta shell), requires an exact state-dict audit, and transfers the completed FP32 predictor to
CUDA once before inference.
Large host-to-CUDA parameter copies are chunked to stay within a Windows/eGPU BAR1 aperture;
the reusable builder remains full FP32 by default. The standalone smoke defaults to explicit
``bfloat16_egpu`` storage for autocast-owned weights while retaining every decoder FFN linear
layer in FP32 because upstream disables autocast there. This qualifies the constrained-device
code path, not full-FP32 numerical parity; ``--weight-storage float32`` remains available for
production-class devices. The constrained smoke also sets SAM grounding and postprocessing
batches to one frame; its session boundary remains exactly 30 frames and Object Multiplex still
fails closed at the builder's 128-object capacity. Its scoped SDPA policy tries Flash, then
memory-efficient attention, then Math so a Windows Torch build without Flash does not abort;
the original upstream backend function is restored during cleanup. The composite arm defaults to a prompt-only
exemplar tile: the exemplar appears beside the target on the prompted frame and is neutral on the
other 29 frames. ``--composite-exemplar-visibility all`` retains the repeated-exemplar diagnostic.

The 2026-09-04 RTX 4090 Laptop qualification exercised more than the legacy smoke. A spawned
production worker built the exact 128-object-capacity `egpu` profile, verified the pinned SAM
tree, streamed 30/30 active frames with no retained dense prediction population, and shut down
without force. Its real `1008:756` east-neighbor relay produced forward/backward seeds; the
forward destination session remained active for 30/30 frames and recovered 88,645 pixels outside
the shared overlap. A separate 59-frame, two-window chain reused one worker/predictor, reinjected
`temporal_dogfood` at the repeated boundary, retained zero prediction masks, and was active on all
59 frames. The single-object golden retained anchor IoU 0.9967405 at 3,636 MiB peak allocation.
Three-object multiplexing reproduced the two tiny-mask anchor diagnostics (81/106 pixels,
IoU 0.8148/0.8396) even when each was run alone, while the 132,958-pixel mask retained 0.9967;
this is small-mask tracker behavior rather than cross-object interference, and exact hard-positive
prompt restoration remains mandatory.

Pinned SAM removes pixels shared by simultaneously injected object masks from each returned
per-object seed preview. Production seed validation therefore remains exact but overlap-aware:
it computes each object's representable exclusive mask, requires at least 95% exclusive support,
and accepts only a byte-exact return of that representable mask and union. Direct diagnostic tools
retain the stricter full-mask-exact policy. Any non-shared erosion, expansion, identity swap, or
high-overlap ambiguity remains fatal; the worker error records the retained seed artifact/hash and
lineage context. The same per-object domain is audited again when propagation model-visits the
prompt; the forward leg owns that visit in a two-leg session and the backward leg owns it in a
backward-only session. That later IoU is diagnostic because qualified tiny masks can be eroded by
the tracker even in isolated sessions. Production discards the model's prompt preview, reinjects
the exact filled seed, and records the diagnostic disposition; original authoritative masks are
also restored after terminal filters.

Before initial, relay, or temporal-dogfood masks enter a multiplex session, production applies a
deterministic first-fit partition over the aggregate shared-pixel domain. A candidate joins the
first session in which every mask still has at least 95% exclusive support and the 128-object cap
is respected; otherwise it opens another session. The worker repeats this check at every window
boundary because independently tracked objects can converge later. Conflicting lineages remain
distinct, retain their complete seed masks, and are unioned only after their separate tracker
sessions; high mask overlap is never treated as proof that two semantic objects are identical.

The local GPU environment currently has an unresolved package-metadata conflict: pinned SAM 3
declares `numpy>=1.26,<2`, while OpenCV 5 and the installed environment use NumPy 2.5.2. The
bounded CUDA runs succeed, but numerical qualification should use a compatible NumPy 1.26 plus
OpenCV 4 environment and compare against H100 FP32 output.

The meta construction option exists for hosts whose commit/pagefile budget cannot hold the
normal CPU shell. It permits exactly the two pinned ViT scalar `linspace(0, 0.1, 32)` schedules
on CPU and rebuilds exactly one nonpersistent 32×32 text causal mask; any other constructor
buffer absent from the checkpoint remains fatal. Meta construction rejects compile and warm-up
until those operations are separately qualified after CUDA materialization. The exact state-dict
and post-transfer device/dtype audits are unchanged.

``python tools/lta_point_smoke.py`` is a separate diagnostic experiment for
SAM 3.1's per-instance point-interactivity path. It compares a distance-transform
point set with a centerline/edge point set, refines only against one known prompt-frame
YOLO polygon, and propagates the final revision through one fixed 30-frame source session.
Its point-seeded score is interaction provenance, not detector confidence, and its
artifacts are not LTA publication outputs. The pinned point-created propagation response
is partial and does not expose Multiplex drop statistics; the diagnostic records that field
as inapplicable while still rejecting any nonzero drop count if a future response supplies it.

``python tools/lta_tile_smoke.py`` compares box and point prompts on a fixed
native-pixel 1008-square crop. ``python tools/lta_mask_seed_smoke.py`` then
exercises a pinned private tracker experiment that seeds the authoritative YOLO mask
directly and advances the shared SAM encoder/tracker one frame at a time. The latter
is deliberately outside the public SAM request API; it is retained because labeled
endpoint checks demonstrated coherent propagation, not as a stable runtime contract.
Its default is the tracker-only branch, and its anchor-integrity and non-anchor-activity
gates are explicitly diagnostic rather than quality or publication acceptance.

From a source checkout, the unprivileged HGX Track-A smoke test is
``python tools/hgx_selftest.py --gpus 4`` (and ``--gpus 8`` for a full-node
allocation). It creates boundary-crossing synthetic objects and requires a byte-identical
GPU/CPU `keep_objects` result. ``--plan-only`` exercises partitioning and row packing on a
host without CUDA. ``python tools/d1_ipc_selftest.py --gpus 4`` verifies that
single-visible-device spawned ranks can export, import, NVLink-OR, and acknowledge the
same dedicated CUDA allocations used by D1 groups; repeat with eight allocated GPUs.
Wheels retain the hardware, LTA, and component-projection diagnostic scripts under
``share/xta/tools``.
Four-GPU representative TTA
qualification completed the full D1 lifetime across model workers, partial publication,
release acknowledgement, and scheduler quiescence; the byte-exact but performance-neutral
result is recorded in the HGX experiment section above.

For v18 specifically, CPU tests cover policy/plan identity, categorical and intensity geometry,
channel/tile primitives, spawn-worker contracts, manifest/ownership safety, and the implemented
CPU-backed `RenderBatch` fan-out. A bounded, nonrepresentative GPU run qualified
basic CUDA/Torch/CuPy execution and the resident renderer: Cartesian and tilted-Cartesian fixtures
met the one-uint8 cross-backend tolerance, while optimized hardware-texture Azimuthal fixtures showed
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

Direct Azimuthal/Tilted rendering and resident mask quantization use 32-by-8 pixel launches,
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

Interpolation represents each endpoint in an odd, centered rectangular canvas with a
background margin. Canvas extents follow the two local shapes; global endpoint travel is
applied only by the world-coordinate painter. Packed component membership uses a compiled
unsigned-word OR/count operation when Numba is available, with the NumPy path retained when
that optional backend is unavailable.

The min-radius acceptance evaluator runs on CPU by default. For a positive rejection
threshold, the smaller endpoint-center SDF can certify a common foreground disk through
every intermediate section. A floating-point margin guards near-threshold certificates;
uncertified plans retain the full section-radius scan. The certificate returns an acceptance
lower bound, while requests without a positive rejection threshold still compute the full
radius. Shape-dependent floating-point EDT rounding can change boundary voxels; the
rectangular implementation does not promise bit-identical output.

`YOLO_TTA_GPU_INTERPOLATION_RADIUS=1` selects the experimental CUDA radius evaluator.
Opted-in planners borrow independent streams and hold the
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
remains one because the structural workspace estimate does not bound every topology-dependent
planner allocation. Per-pass
logs and runtime stats separately report radius/render backends, autotune timings, lock wait,
execution time, transfer categories, crop/patch pixels, cache eviction, fallback, and the
worker-visible physical CUDA token.

## Streaming inference completion and terminal fusion

### Component projection replay

`--capture_component_replay PERSISTENT_DIR` copies a bounded sample of immutable
view-native Azimuthal components, output geometry and checksums during TTA. The default
selects three vertical +30-degree tilted-Azimuthal views, one component each, within a
4 GiB total input budget. `--capture_component_views` and
`--capture_component_limit` change the selection and count. The capture directory
must survive the job; cluster `/tmp` is unsuitable. Capture is disabled by default.

`python tools/replay_component_projection.py CAPTURE_DIR --output RESULT_DIR`
compares the reference and sparse projectors in fresh CPU processes and checks
decoded output slices without repeating inference. Its local temporary workspaces
are removed after completion; logs and metrics remain in the output directory.
`--cuda-reference` explicitly enables a CUDA reference attempt and records whether
an eligible GPU path actually ran. Installed wheels place this tool under
`share/xta/tools`.

### Completion ownership

OpenVINO request callbacks publish indexed completions into a bounded queue. A bounded
consumer pool, sized to the useful infer-request count, performs output decoding and
destination writes concurrently. Destination slices and aggregate statistics have separate
locks, callback/request draining is unconditional on failure, and competing failures are
reported in submission order so concurrency does not make error selection nondeterministic.

The scheduler treats each runtime TTA view as terminal when its full-frame/tile continuation
has retired. As soon as every variant of one physical view is terminal, ownership of that
group is detached from the inference registries, its variants are OR-collapsed, and any
Azimuthal/Tilted projection runs while other views may still be inferencing. Completed physical
views feed one path-backed, single-writer source-space union reducer. Equal-geometry sparse
component layers are ORed before one restore per output slice, retaining the grouped G5
optimization. A one-credit dense handoff prevents finalizers from retaining multiple
source-sized volumes while waiting for the reducer.

Finalization and reducer futures are first-class scheduler dependencies, including terminal
coverage assertions at quiescence. The global centerline/smoothing stages still wait for the
complete union because their semantics span every view, but they no longer wait for a
separate post-inference collapse/backprojection/fusion phase.
