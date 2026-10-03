# Security runtime migration validation

## Decision and scope

The default old-compatible **macOS ARM64 CPU float32** runtime passes every
threshold frozen in `criteria.json` before the first old/new comparison.
Python 3.12.15, PyTorch 2.14.1, Lightning 2.6.6 and 96 pinned PyPI packages
replace the historical exported environment. OSV matches no advisory for the
96 selected versions on 2026-10-03. This query excludes Python, native system
libraries and the local runtime; it is not an application-wide security guarantee.
The same name/version query matched advisories for 39 of 166 recognized Python
package versions in the original export. Conda downstream patches were not verified.

The original checkpoint and molecular inputs are unchanged. See
`reference-provenance.json` for hashes. No model retraining was used to replace
its learned weights. The reference commit is `da22371` (after the GitPython PR).

## Independent reference

The reference ran the original source with Python 3.10.19, PyTorch 2.0.1,
Lightning 1.7.4, NumPy 1.22.3, SciPy 1.7.3 and BioPython 1.79 on macOS ARM64.
RDKit 2022.9.5 and matplotlib 3.5.3 substitute the Linux Conda versions
2022.03.2 and 3.4.3. Python's patch version and several transitive packages also
differ. `legacy-pins.txt` records the reconstructed reference, not the author's
exact Linux/CUDA training environment.

The old torch-scatter 2.1.1 native extension fails to compile with current Apple
clang/PyTorch 2.0.1 headers. The original, unchanged `scatter.py` and `utils.py`
were loaded without native package initialization. DiffInt uses only add/mean;
those original functions call native PyTorch `scatter_add_`, not the extension.
Their source hashes are recorded. The maintained runtime implements these
narrow batch reductions without torch-scatter/torch-geometric dependencies.

## Frozen comparisons

Input: the repository's `1a2g_A_rec.pdb` / `.sdf`, 8 Å pocket selection and ODDT
hydrogen-bond particles. All coordinates, masks, one-hot arrays and interaction
IDs match exactly. A batch of four copies contains 40 ligand nodes and 140 pocket
nodes, with different fixed test features. CPU uses one thread.

- Dynamics at t = 0, 0.1, 0.5, 0.9, 1: max absolute output difference
  **7.62939453125e-6**, below 1e-5.
- Training loss at seeds 1729, 19, 73: **exact**.
- All recorded first-order parameter gradients: max difference
  **3.427267074584961e-7**, below 1e-4.
- One original AdamW/AMSGrad step (lr 1e-3): max weight difference
  **1.3984739780426025e-5**, below 1e-4.
- Three samples from the same pocket, 12/16/20 ligand nodes, fixed seed 1729,
  full 500 diffusion steps: all atom types and batch masks match; maximum ligand
  coordinate difference **8.993595838546753e-5 Å**, below 1e-3; pocket difference
  **1.3709068298339844e-5 Å**.

The three generated molecules also retain identical bond graphs, sanitized
SMILES and SA scores after molecule processing/UFF relaxation. The maximum QED
difference is 5.551115123125783e-17 (`chemistry-comparison.json`).

`comparison.json` records full-array maxima. The compact baseline stores all
forward/loss results and 2,048 regularly spaced elements per gradient/weight
array. Full arrays were compared locally and retained in the maintenance workspace.
This is one real input pocket, not the complete CrossDock evaluation set.

## Why compatibility math is needed

Standard current PyTorch math produced max output difference 2.86102294921875e-5,
exceeding the frozen 1e-5 criterion. Its generated atom types still matched,
but standard mode is not claimed to pass that strict numerical criterion.
It remains selectable with `--no-old-compatible`.

The historical macOS build did not enable `AT_BUILD_ARM_VEC256_WITH_SLEEF`, while
the new build does. Same-input SiLU probes differed by at most 9.5367431640625e-7.
The old scalar expf expression matched the independent reference grid exactly.
Same old inputs passed to every recorded Linear layer gave identical results.
After activations were restored, `torch.cross` still differed by 3.814697265625e-6
on identical vectors. Explicit separate products/subtractions match the old cross
output. Together these changes bring the full default comparison below the
unchanged thresholds; no exception to the thresholds was needed.

The production compatibility implementation calls only system libm expf/tanhf
through a minimal buffer-checked C extension, and uses current PyTorch operators
for the analytical first derivatives and cross product. It does not load old
PyTorch code, use global monkeypatches or ship the diagnostic dylib.
Higher-order gradients, AMP and non-CPU float32 are rejected in compatibility mode.
Bitwise equality across all shapes, CPUs and operating systems is not promised.

Primary implementation references:
[PyTorch 2.0.1 activation kernels](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/Activation.cpp),
[historical ARM vector math](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h).

## Runtime and security checks

- Nine automated tests (six runtime and three Git metadata/discovery tests): batch reduction/gradients; restricted checkpoint roundtrip
  and unsupported pickle rejection; NumPy object rejection; 20 standard amino-acid
  mappings; old activations/first derivatives and C buffer validation; both model
  modes saved/reloaded and overridden with identical weights.
- Both modes: actual `train.py` CLI, one epoch on the real input fixture,
  offline WandB metrics, checkpoint saved, optimizer/loop state resumed through
  a second epoch. Both reach global step 2 and retain the requested mode and 130
  optimizer parameter states. Checkpoint callback epoch metadata is captured before
  or after loop completion depending on the save event; actual step/processed-epoch
  counts were checked rather than assuming the metadata convention.
- Actual `test_single.py` CLI: pocket preprocessing, hydrogen-bond generation,
  default compatible 500-step generation, molecule processing and SDF saving.
  Previously ignored `--n_samples` / `--batch_size` arguments are now honored.
- Tensor checkpoint loading explicitly uses `weights_only=True`, with only legacy
  Namespace/path configuration types allowlisted. Newer Lightning hooks and gradient
  clipping signatures are used; initialization is on CPU before the caller/Trainer
  moves the full model. NumPy object-array inputs are rejected.
- NumPy 2.3.5 is intentional: ODDT 0.7 fails on removed `np.in1d` in newer NumPy.
  It passes the recorded advisory query and preserves the fixture's hydrogen bonds.
- Fresh environment installation and `uv pip check` cover the complete lock and
  local compiled runtime. The notebook's renderer is checked locally; Google Colab
  itself and its currently supplied Python version are not validated.

## Remaining limits

CUDA/GPU, Linux runtime, complete data preparation/evaluation, 1,000-epoch training,
periodic training visualizations/large sampling, distributed training and real
smina/AutoDock/Vina docking were not validated. Docking executables are external
requirements, not supplied by the Python lock. The historical CUDA export is
archived for provenance, not recommended as a maintained environment.

The bundled SA fragment table is still trusted pickle data; the code is not
intended to accept an untrusted replacement. Legacy object NPZ files require an
explicit reviewed conversion outside the normal loader. No automatic unpickling
fallback was introduced.

No GitHub Actions workflow is added: the current OAuth token lacks `workflow`
scope (observed during the preceding Mothra update). CI execution is not claimed.
