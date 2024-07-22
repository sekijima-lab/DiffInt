## Dependencies

### Cuda environment
| Software     | Version |
|--------------|---------|
| CUDA         | 11.8    |
| cudnn        | 8.9.7   |

### conda environment
```bash
conda env create -n Int-env -f environment.yml
```

| Software          | Version   |
|-------------------|-----------|
| Python            | 3.10.4    |
| numpy             | 1.22.3    |
| PyTorch           | 2.0.1     |
| PyTorch cuda      | 11.8      |
| Torchvision       | 0.15.2    |
| Torchaudio        | 2.0.2     |
| PyTorch Scatter   | 2.1.1     |
| PyTorch Lightning | 1.7.4     |
| RDKit             | 2022.03.2 |
| WandB             | 0.13.1    |
| BioPython         | 1.79      |
| imageio           | 2.21.2    |
| SciPy             | 1.7.3     |
| OpenBabel         | 3.1.1     |
| ODDT              | 0.7       |


### Data download
Download the training, validation and test datasets: [Data](https://drive.google.com/file/d/1RwDXBRVLRcEjSNHTw1JG6TpNgNUIogX2/view?usp=sharing)

```bash
tar xvzf DiffInt_crossdock_data.tar.gz
```

### Data construction by yourself
(You don't need to construct data by yourself.)
Download and extract the dataset as described by the authors of [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol/tree/main/data)
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
python process_crossdock.py data/ --outdir ./data/crossdocked_ca/
```

data preparation step 2: add hydrogen bonds information
```bash
python hbond_double.py --data_dir /step_1/directory/path/ --out_dir /step_2/directory/path// --pdb_dir /pdb_data/directory/path/
```
For example
```bash
python hbond_double.py --data_dir ./data/crossdock_ca/ --out_dir ./data/crossdocked_ca_Int/ --pdb_dir ./data/crossdocked_pocket10/
```

### Training
```bash
python -u train.py --config config/DiffInt_ca_double.yml
```

### Molecule generation
Generation of 100 ligand molecules for 100 protein pockets.

```bash
python --checkpoint checkpoint_file --test_dir /data/directory/path/ --outdir /out/directory/path/
```
For example
```bash
python test_npz.py --checkpoint checkpoints/best_model.ckpt --test_dir DiffInt_crossdock_data/ --outdir sample
```

Download generated molecules: [sdf_files](https://drive.google.com/file/d/1c0QSldeYq7mVF7_iGiJHKV7IiRxfi9mA/view?usp=sharing)

### Generate 100 ligand molecules for one pocket
You can use Google Colabratory

```bash
.
├── colab
│   └── DiffInt_generate.ipynb
```
