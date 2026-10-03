# Runtime validation

Install the isolated runtime from the top-level README, including the local C extension.
Run all tests on Python 3.12.15:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pip check
```

The runtime tests exercise numerical activations and first derivatives, model modes,
restricted checkpoint loading, numeric-only datasets and batch reductions.
The retained GitPython tests exercise the current WandB Git client using only inert
local repositories. They require no WandB account or network connection.

See [the current validation report](../validation/REPORT.md) for CPU comparisons and
limitations. The earlier isolated GitPython result is retained as a historical
record in `validation/legacy-gitpython-report.md`; its old dependency pins are archived
as Markdown rather than an active installation file.
