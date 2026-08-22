# Volume TTA architecture

The `GPT-5.6-Sol-Pro_v17.0.10_SLURM.py` filename remains a versioned compatibility
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

The physical split preserves the original numerical functions. Callable-only references
that were legal forward references in the monolith are resolved by `_latebind`; imports
needed for decorators, class construction, defaults or module initialization remain normal
eager imports. `_latebind` is transitional compatibility infrastructure for the physical
split, not a service locator for new code. New code should use explicit one-way imports.

## Backend boundary

`volume_tta.inference_backends` contains dependency-free control-plane contracts. The
current CUDA/OpenVINO scheduler has not been rewritten in this refactor; its tuned leasing
and hybrid policy remains behaviorally unchanged. Future scheduler work should adapt those
workers to `InferenceBackend` rather than adding another backend-specific branch to
`pipeline.main()`.

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

The refactor manifest records the AST digest of all 1,133 historical top-level executable
statements. The original monolith was intentionally retired after the manifest was
checked in; `tools/verify_refactor.py` now verifies the package against that immutable
inventory without requiring the deleted source file. Reviewed post-refactor seams are
listed explicitly in the verifier rather than hidden by regenerating the manifest. The
test suite also covers dependency-light CLI/config imports, configuration parsing,
cycle-safe package imports, accelerator policy/lifecycle, and collective-ready backend
contracts.

Run the dependency-light checks with:

```powershell
python -m unittest discover -s tests -v
python tools/verify_refactor.py
python tools/smoke_import.py
```

Hardware-backed CUDA, TensorRT, OpenVINO, QAT, IAA, DSA and full data/model parity tests
still require the production environment and representative artifacts. Intel accelerator
build, provisioning, and admission instructions live in ``native/README.md``; run
``python tools/intel_accelerator_selftest.py --backend all`` on the target host.
