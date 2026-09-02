# External augmentation profiles

Each GPU file is self-contained and exports the
`build_gpu_augmentation(device=..., batch_size=...)` factory expected by
XTA PTA's offline GPU backend. `GPU_baseline.py` is the cleaned copy of the
supplied GPU policy; references to the "standard" policy in the profile request
mean this baseline.

Each corresponding `CPU_*.py` file exports `build_augmentation()` for PTA's
offline CPU backend. The CPU policies use the same transform-selection graph,
profile magnitudes, and integer-seed contract as their GPU counterparts.

Both four-profile families keep the same transform-selection probabilities and
the same random draw order. Only sampled magnitudes change, so a fixed integer
seed selects the same D4 element, scale branch, elastic/brightness/blur
switches, noise family, and salt-and-pepper switch in every profile and on both
backends.

| Parameter | Light | Baseline | Heavy | Super-heavy |
| --- | ---: | ---: | ---: | ---: |
| Rotation | ±35° | ±45° | ±55° | ±70° |
| Scale | 0.667–1.5× | 0.5–2× | 0.4–2.5× | 0.333–3× |
| Translation per axis | ±7.5% | ±10% | ±12.5% | ±17.5% |
| Shear per axis | ±22.5° | ±30° | ±37.5° | ±42° |
| Elastic displacement amplitude | 15 px | 20 px | 27.5 px | 35 px |
| Brightness multiplier | 0.85–1.15× | 0.8–1.2× | 0.75–1.25× | 0.65–1.35× |
| Gaussian blur sigma | 0–3.5 | 0–5 | 0–6.5 | 0–8 |
| Additive Gaussian noise sigma | 0–0.35 | 0–0.5 | 0–0.65 | 0–0.85 |
| Shot-noise strength | 0–0.035 | 0–0.05 | 0–0.065 | 0–0.085 |
| Multiplicative noise | 0.65–1.35× | 0.5–1.5× | 0.35–1.65× | 0.15–1.85× |
| Salt-and-pepper amount | 0–3.5% | 0–5% | 0–6.5% | 0–8.5% |

D4 rotation/reflection remains uniformly selected in all profiles. Elastic
deformation remains active for 30% of augmented copies, brightness for 50%,
blur for 25%, and salt-and-pepper for 25%. One of the three primary noise
families is selected uniformly for every augmented copy. The super-heavy shear
limit stays below 45° to keep the composed two-axis shear away from its singular
endpoint.

For example:

```text
python -m XTA --mode pta --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY --augmentation XTA/examples/external_augmentations/GPU_baseline.py --augmentation_ratio 4 --augmentation_execution offline --offline_augmentation_backend gpu
```

Set `PTA_GPU_TORCH_COMPILE=0` to disable optional compilation of the fused
pointwise kernel. The GPU policies require a CUDA-enabled PyTorch installation.
The hardware-gated contract test for all four profiles is:

```text
XTA_RUN_EXTERNAL_AUGMENTATION_CUDA=1 python -m unittest tests.test_external_augmentation_examples -v
```

The CPU policies are distribution-compatible rather than pixel-identical to
their GPU counterparts because OpenCV/NumPy and CUDA use different resamplers
and random-number streams. Select one with, for example:

```text
python -m XTA --mode pta --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY --augmentation XTA/examples/external_augmentations/CPU_baseline.py --augmentation_ratio 4 --augmentation_execution offline --offline_augmentation_backend cpu
```
