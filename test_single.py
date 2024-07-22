import argparse
import warnings
from pathlib import Path
from time import time
import numpy as np
import os

import torch
from rdkit import Chem
from tqdm import tqdm
import oddt

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule
from dataset import ProcessedLigandPocketDataset
import utils
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.conditional_model import ConditionalDDPM
from torch_scatter import scatter_add, scatter_mean
from process_crossdock import process_ligand_and_pocket
from hbond_double import hbond_create


def ligand_generation(test_file, checkpoint, outdir=None, batch_size=120, n_samples=100, relax=True, all_frags=True, save=True):
    #t_pocket_start = time()      
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_dims = 3
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device)

    test_dataset = ProcessedLigandPocketDataset(test_file, center=False)

    if save:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        raw_sdf_dir = Path(outdir, 'raw')
        raw_sdf_dir.mkdir(exist_ok=True)
        processed_sdf_dir = Path(outdir, 'processed')
        processed_sdf_dir.mkdir(exist_ok=True)
        sdf_out_file_raw = Path(raw_sdf_dir, test_dataset[0]['names']+'_gen.sdf')
        sdf_out_file_processed = Path(processed_sdf_dir, test_dataset[0]['names']+'_gen.sdf')

    pocket_coords = test_dataset[0]['pocket_coords'].repeat(batch_size,1)
    pocket_one_hot = test_dataset[0]['pocket_one_hot'].repeat(batch_size,1)
    pocket_mask = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=INT_TYPE),len(test_dataset[0]['pocket_mask'])
        )
    pocket_size = torch.tensor([test_dataset[0]['num_pocket_nodes']]*batch_size, device=device, dtype=INT_TYPE)

    pocket ={
        'x': pocket_coords.to(device, FLOAT_TYPE),
        'one_hot': pocket_one_hot.to(device, FLOAT_TYPE),
        'size': pocket_size.to(device, INT_TYPE),
        'mask': pocket_mask.to(device, INT_TYPE)
        }
    
    # Pocket's center of mass
    pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

    num_nodes_lig = None
    if num_nodes_lig is None:
        num_nodes_lig = model.ddpm.size_distribution.sample_conditional(n1=None, n2=pocket['size'])

    # statistical data
    all_molecules = []
    valid_molecules = []
    processed_molecules = []
    n_generated = 0
    n_valid = 0
    iter=0

    while len(valid_molecules) < n_samples:
        iter +=1

        if type(model.ddpm) == ConditionalDDPM: 
            print(type(model.ddpm))
            xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.sample_given_pocket(pocket, num_nodes_lig,timesteps=None)

        pocket_com_after = scatter_mean(xh_pocket[:, :x_dims], pocket_mask, dim=0)
        xh_pocket[:, :x_dims] += \
                 (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :x_dims] += \
                (pocket_com_before - pocket_com_after)[lig_mask]
        
        # Build mol objects
        x = xh_lig[:, :x_dims].detach().cpu()
        atom_type = xh_lig[:, x_dims:].argmax(1).detach().cpu()
        lig_mask = lig_mask.cpu()
        
        dataset_info = dataset_params['crossdock_h']

        mols_batch = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                            utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                    add_hydrogens=False,
                                    sanitize=True,
                                    relax_iter=0,
                                    largest_frag=False)
            if mol is not None:
                mols_batch.append(mol)

        all_molecules.extend(mols_batch)
        # Filter to find valid molecules
        mols_batch_processed = [
                process_molecule(m, sanitize=True,
                                relax_iter=(200 if relax else 0),
                                largest_frag=not all_frags)
                for m in mols_batch
            ]
        processed_molecules.extend(mols_batch_processed)
        valid_mols_batch = [m for m in mols_batch_processed if m is not None]
        
        n_generated += batch_size
        n_valid += len(valid_mols_batch)
        valid_molecules.extend(valid_mols_batch)

    valid_molecules = valid_molecules[:n_samples]

    # Reorder raw files
    all_molecules = \
        [all_molecules[i] for i, m in enumerate(processed_molecules)
        if m is not None] + \
        [all_molecules[i] for i, m in enumerate(processed_molecules)
        if m is None]

    if save:
        # Write SDF files
        utils.write_sdf_file(sdf_out_file_raw, all_molecules)
        utils.write_sdf_file(sdf_out_file_processed, valid_molecules)

    # Time the sampling process
    #time_per_pocket[str(sdf_file)] = time() - t_pocket_start
    #with open(time_file, 'w') as f:
    #    f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")
    return valid_molecules

def process_data(sdf_file, pdb_file):
    dataset_info = dataset_params['crossdock']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    dist_cutoff=8.0
    ligand_data, pocket_data = process_ligand_and_pocket(pdb_file,
                                                          sdf_file,
                                                          atom_dict=atom_dict, dist_cutoff=dist_cutoff,amino_acid_dict=amino_acid_dict, ca_only=True)

    lig_mask = np.zeros(len(ligand_data['lig_coords']))
    pocket_mask = np.zeros(len(pocket_data['pocket_coords']))
    file_name = np.array([Path(pdb_file).stem])

    npz_name = 'single1.npz'
    np.savez(npz_name,
        names = file_name,
        lig_coords=ligand_data['lig_coords'],
        lig_one_hot=ligand_data['lig_one_hot'],
        lig_mask = lig_mask,
        pocket_coords=pocket_data['pocket_coords'],
        pocket_one_hot=pocket_data['pocket_one_hot'],
        pocket_mask = pocket_mask
        )
    
    return npz_name
    
def process_data_h(sdf_file, pdb_file, npz_name):
    test_npz = np.load(npz_name)
    protein = next(oddt.toolkit.readfile('pdb',pdb_file))
    ligand = next(oddt.toolkit.readfile('sdf',sdf_file))
    protein.protein = True
    ligand.removeh()

    int_id, inter_coords, inter_one_hot = hbond_create(protein, ligand)

    po_one_tmp1 = np.append(test_npz['pocket_one_hot'],np.zeros((len(test_npz['pocket_one_hot']),2)),axis=1)

    if len(inter_one_hot)>0:
        po_int_cor_tmp = np.append(test_npz['pocket_coords'],inter_coords,axis=0)
        inter_one_hot = np.append(np.zeros((len(inter_one_hot),20)),inter_one_hot,axis=1)
        po_int_one_tmp = np.append(po_one_tmp1,inter_one_hot,axis=0)
    else:
        po_int_cor_tmp = test_npz['pocket_coords']
        po_int_one_tmp = po_one_tmp1

    pocket_mask = np.append(test_npz['pocket_mask'], np.zeros(len(inter_one_hot)))
    inter_mask = np.zeros(len(inter_one_hot))

    #buffer2 = io.BytesIO()
    test_npz_h = 'test.npz'
    np.savez(test_npz_h,
        names = test_npz['names'],
        lig_coords=test_npz['lig_coords'],
        lig_one_hot=test_npz['lig_one_hot'],
        lig_mask = test_npz['lig_mask'],
        pocket_coords = po_int_cor_tmp,
        pocket_one_hot= po_int_one_tmp,
        pocket_mask = pocket_mask,
        inter_id = int_id,
        inter_mask = inter_mask
        )

    return test_npz_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,default=None)
    parser.add_argument('--test_file', type=Path,default=None)
    parser.add_argument('--test_dir', type=Path,default=None)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--fix_n_nodes', action='store_true')
    parser.add_argument('--n_nodes_bias', type=int, default=0)
    parser.add_argument('--n_nodes_min', type=int, default=0)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    
    ligand_generation(outdir=args.outdir, test_file=args.test_file, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()