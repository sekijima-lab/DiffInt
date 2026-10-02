# GitPython security update validation

GitPython 3.1.31 is replaced by 3.1.62 for reviewed security advisories,
including [CVE-2026-87817 / GHSA-239g-whfq-7xj9](https://github.com/advisories/GHSA-239g-whfq-7xj9).
Tracked repository content must not be mistaken for the real `.git` directory.
Opening a malicious repository could otherwise select attacker-controlled
configuration or hooks. This does not imply that DiffInt has been exploited.

The conda-forge 3.1.62 build requires Python >=3.11. The official PyPI wheel
supports Python >=3.7, so only GitPython moves to the existing `pip:` section.
Python, PyTorch, CUDA, WandB and all other runtime pins are unchanged.

## Reproduce the scoped checks

Use a **new** Python 3.10 virtual environment; do not modify the training environment.

```sh
python -m pip install -r tests/requirements-gitpython.txt
python tests/test_gitpython_compatibility.py
python -m pip check
```

The real WandB 0.13.1 `GitRepo` client is exercised, without a WandB account or
network calls: commit, branch, email, remote URL, dirty state, untracked files,
diff, local cloning, bare repositories, and linked worktrees. A fixture uses
inert tracked `HEAD`, `objects`, `refs`, `gitdir`, `commondir` and `config` files
to verify the real `.git` directory and configuration win. No executable hook
or external configuration include is created.

Validated on macOS arm64 / Python 3.10.19. The two compatibility tests pass
with both 3.1.31 and 3.1.62, holding the other 20 validation dependencies
identical. The security discovery check fails on 3.1.31 and passes on 3.1.62.
All three checks and `uv pip check` pass after the update. An OSV query for
3.1.62 returned no advisories on 2026-10-02.

For the baseline comparison, change only `gitpython==3.1.62` to
`gitpython==3.1.31` in a second environment and run with
`GITPYTHON_BASELINE=1` to skip the intentionally failing security check.
Run the security check without that flag to reproduce the pre-fix failure.

## Limits

This is a Git metadata/discovery test environment, not a replacement for the
full Linux/CUDA environment. Full environment solving, GPU training,
pretrained-model inference, docking, WandB server communication and all
remaining dependency advisories are outside this PR. The requirements retain
old dependencies to isolate this change; they are not a general secure lockfile.
