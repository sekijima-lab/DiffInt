
# DiffInt: A Diffusion Model for Structure-Based Drug Design with Explicit Hydrogen Bond Interaction Guidance

This repository provides the source code and pretrained models for the [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01385).

DiffInt is a diffusion-based generative model that explicitly incorporates hydrogen bond interactions for structure-based drug design. It introduces interaction particles to guide ligand generation within protein binding pockets.


## Installation and runtime

The maintained runtime is Python **3.12.15** with PyTorch **2.14.1**, PyTorch
Lightning **2.6.6**, NumPy **2.3.5**, RDKit **2026.3.6**, BioPython **1.88** and
WandB **0.30.0**. `requirements-lock.txt` pins the resolved packages.
NumPy remains at 2.3.5 because ODDT 0.7 uses `np.in1d`, removed in NumPy 2.4.

Create a new environment instead of changing an existing research environment.
With a Python 3.12.15 interpreter and `uv` installed:

```bash
uv --no-config venv --python /path/to/python3.12.15 .venv
source .venv/bin/activate
uv --no-config pip install -c requirements-lock.txt -r requirements-build.txt
uv --no-config pip install --no-build-isolation -r requirements.txt
uv --no-config pip install --no-build-isolation --no-deps -e .
uv --no-config pip check
python -m unittest discover -s tests -p test_runtime.py -v
```

ODDT imports dependencies while building; the first installation step supplies
those dependencies. The local runtime builds a small C extension with a C
compiler (`clang`/`gcc`) and the Python development headers. It uses the system's
`expf`/`tanhf`; it does not link an old PyTorch library.
Open Babel is provided by `openbabel-wheel` inside this environment.

Default **`--old-compatible`** mode uses CPU float32, scalar libm activations and
explicit cross-product arithmetic to reduce differences from the historical
CPU reference. **`--no-old-compatible`** selects standard current PyTorch math.
The mode is saved in checkpoints and training `runtime.json`, and is printed
when a model is created. The compatibility mode supports first-order gradients;
it rejects mixed precision and higher-order gradients.

[Validation results and limitations](validation/REPORT.md) cover macOS ARM64 CPU.
The original Linux/CUDA environment is archived in
[the historical environment record](validation/legacy-environment.md).
CUDA/GPU, full training, full dataset evaluation and real docking have not been
validated in this migration. To experiment with GPU execution, explicitly use
standard math and set training `accelerator: gpu`; this is outside the measured scope.
The obsolete Linux-specific `environment.yml` is replaced by the maintained CPU
requirements, not a claim of a verified CUDA upgrade.

Checkpoint loading uses `weights_only=True` with a small allowlist for legacy
configuration types. NPZ datasets must contain numeric/string arrays; object
arrays are rejected rather than implicitly unpickled. A legacy object dataset
requires a separate, reviewed conversion before use. The bundled SA fragment
table is a trusted pickle; do not replace it with untrusted content.

### Data download
Download the training, validation and test datasets: [Data](https://drive.google.com/file/d/1RwDXBRVLRcEjSNHTw1JG6TpNgNUIogX2/view?usp=sharing)

```bash
tar xvzf DiffInt_crossdock_data.tar.gz
```

### Data construction by yourself
(You don't need to construct data by yourself.)
Download and extract the dataset as described by the authors of [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol/tree/main/data).  
Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` to `data` directory.
```bash
.
├── data
│   ├── DiffInt_crossdock_data.tar.gz
│   └── split_by_name.pt
```
Extract the TAR archive
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```

data preparation step 1
```bash
python process_crossdock.py /data/directory/path/ --outdir /output/directory/path/
```
For example
```bash
python process_crossdock.py data/ --outdir data/crossdocked_ca/
```

data preparation step 2: add hydrogen bonds information
```bash
python hbond_double.py --data_dir /step_1/directory/path/ --out_dir /step_2/directory/path/ --pdb_dir /pdb_data/directory/path/
```
For example
```bash
python hbond_double.py --data_dir data/crossdock_ca/ --out_dir data/crossdocked_ca_Int/ --pdb_dir data/crossdocked_pocket10/
```

### Training
```bash
python -u train.py --config configs/DiffInt_ca_double.yml
```

### Molecule generation
Generation of 100 ligand molecules for 100 protein pockets.

```bash
python test_npz.py --checkpoint checkpoint_file --test_dir /data/directory/path/ --outdir /out/directory/path/
```
For example
```bash
python test_npz.py --checkpoint checkpoints/best_model.ckpt --test_dir DiffInt_crossdock_data/ --outdir sample
```

Generated molecules used in the paper are ```example/DiffInt_generated_molecules.tar.gz ```


### Generate 100 ligand molecules for one pocket
```bash
python test_single.py --checkpoint checkpoint_file --outdir /out/directory/path/ --pdb /pdb/file/path/ --sdf /sdf/file/path/
```

The Colab notebook is an isolated-environment template requiring Python 3.12.15 or later. Its setup and renderer use a separate interpreter; execution on Google Colab has not been validated for this migration.

```bash
.
├── colab
│   └── DiffInt_generate.ipynb
```
