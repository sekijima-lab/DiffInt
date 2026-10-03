"""Compare a benchmark output directory with the committed compact CPU baseline."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
a = p.parse_args()
base = Path(__file__).resolve().parent
criteria = json.loads((base / 'criteria.json').read_text())
results = {}
for filename, reference_name in [('input.npz', 'legacy-input.npz'),
                                  ('numeric.npz', 'legacy-numeric-sample.npz'),
                                  ('sampling.npz', 'legacy-sampling.npz')]:
    with np.load(a.output / filename, allow_pickle=False) as current, np.load(base / reference_name, allow_pickle=False) as old:
        for key in old.files:
            if key.endswith('_indices'):
                continue
            x, y = old[key], current[key]
            if key + '_indices' in old.files:
                y = y.reshape(-1)[old[key + '_indices']]
            assert x.shape == y.shape, (filename, key, x.shape, y.shape)
            if filename == 'input.npz' or key == 'mask':
                assert np.array_equal(x, y), (filename, key)
                continue
            assert np.isfinite(x).all() and np.isfinite(y).all(), key
            if filename == 'sampling.npz':
                if key == 'ligand':
                    assert np.array_equal(x[:, 3:].argmax(1), y[:, 3:].argmax(1)), 'atom types'
                x, y = x[:, :3], y[:, :3]
                limit = criteria['sampling_coordinates_max_abs']
            else:
                name = ('forward_max_abs' if key.startswith('forward_') else
                        'loss_max_abs' if key.startswith('loss_') else
                        'gradient_max_abs' if key.startswith('grad_') else
                        'one_adamw_step_weights_max_abs')
                limit = criteria[name]
            difference = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            assert difference <= limit, (key, difference, limit)
            results[key] = difference
print(json.dumps({'verdict': 'PASS', 'scope': 'Compact gradient/weight sample; full forward/preprocessing/generation', 'max_abs': results}, indent=2))
