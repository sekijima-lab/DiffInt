# Reproduce the CPU checks

Install the new isolated runtime as described in the top-level README. Use a C
compiler and Python 3.12.15. Keep the original environment unchanged.

```bash
python -m unittest discover -s tests -p test_runtime.py -v
python validation/benchmark.py --repo "$PWD" --out /tmp/diffint-new --sample
python validation/compare.py /tmp/diffint-new
```

Compare `numeric.npz` to the compact reference by selecting the recorded
`*_indices` for gradient/weights arrays; forward/loss arrays are stored in full.
Compare `input.npz` and `sampling.npz` against the corresponding `legacy-*.npz`.
Use the unchanged thresholds in `criteria.json`; atom types are the argmax of
sampling ligand columns after the three position columns.

To regenerate the independent old reference, extract commit `da22371` into a
separate directory and create a Python 3.10.19 environment with `legacy-pins.txt`.
Lightning 1.7.4 has invalid historical `torch>=1.9.*` metadata: install it with
pip 24.0 in this reference environment, not a recent pip. Install the scientific
packages before building ODDT with build isolation disabled. Activate the reference
so its Open Babel executable is on PATH. The old environment has known advisories;
use it only for offline controlled reference inputs.

Download/unpack the source distribution of torch-scatter 2.1.1, and set
`DIFFINT_LEGACY_SCATTER_SOURCE` to its `torch_scatter` directory. Verify
`scatter.py`/`utils.py` against the hashes in `reference-provenance.json`.
Then run this repository's benchmark script with the old interpreter:

```bash
DIFFINT_LEGACY_SCATTER_SOURCE=/path/to/torch_scatter \
  /path/to/reference/bin/python validation/benchmark.py \
  --repo /path/to/original-DiffInt --out /tmp/diffint-old --legacy --sample
```

The bootstrap bypasses native extension initialization but loads the unchanged
original add/mean functions. It is a reference-only helper, not a runtime dependency.
Full gradient/weight arrays need several dozen MB and are intentionally separate
from the compact baseline committed here. Seeds, timestep grid and 500-step
sampling are explicit in the script.

For actual offline training/optimizer-resume smoke in both modes, using the
committed real input fixture and a new empty output directory:

```bash
python validation/cli_smoke.py --out /tmp/diffint-cli-smoke
```

This runs one batch per epoch, two epochs per mode, with periodic large sampling
and visualization disabled. It verifies optimizer state, processed epochs and
saved mode; it does not establish full-training numerical equivalence.
