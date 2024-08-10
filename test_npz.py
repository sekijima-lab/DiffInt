import argparse
import warnings
from pathlib import Path
from time import time

import torch
from rdkit import Chem
from tqdm import tqdm

from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule
from dataset import ProcessedLigandPocketDataset
import utils
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.conditional_model import ConditionalDDPM
from torch_scatter import scatter_add, scatter_mean

if __name__ == "__main__":
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

    start = time()
    print('start time: ', start)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x_dims = 3

    args.outdir.mkdir(exist_ok=args.skip_existing)
    raw_sdf_dir = Path(args.outdir, 'raw')
    raw_sdf_dir.mkdir(exist_ok=args.skip_existing)
    processed_sdf_dir = Path(args.outdir, 'processed')
    processed_sdf_dir.mkdir(exist_ok=args.skip_existing)
    times_dir = Path(args.outdir, 'pocket_times')
    times_dir.mkdir(exist_ok=args.skip_existing)

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)
    
    test_dataset = ProcessedLigandPocketDataset(
                Path(args.test_dir, 'test.npz'), center=False)
    
    num_nodes_lig = None
    pbar = tqdm(test_dataset)
    time_per_pocket = {}
    for d in pbar:
        t_pocket_start = time()
        pdb_file = Path(d['names'].split('.pdb_')[0].split('/')[1].replace('_','-'))
        sdf_file = Path(d['names'].split('.pdb')[1].split('/')[1]).stem.replace('_','-')
         
        sdf_out_file_raw = Path(raw_sdf_dir, f'{pdb_file}_'f'{sdf_file}_gen.sdf')
        sdf_out_file_processed = Path(processed_sdf_dir, f'{pdb_file}_'f'{sdf_file}_gen.sdf')
        time_file = Path(times_dir, f'{pdb_file}_'f'{sdf_file}.txt')
 
        pocket_coords = d['pocket_coords'].repeat(args.batch_size,1)
        pocket_one_hot = d['pocket_one_hot'].repeat(args.batch_size,1)
        pocket_mask = torch.repeat_interleave(
            torch.arange(args.batch_size, device=device, dtype=INT_TYPE),len(d['pocket_mask'])
        )
        pocket_size = torch.tensor([d['num_pocket_nodes']]*args.batch_size, device=device, dtype=INT_TYPE)

        pocket ={
            'x': pocket_coords.to(device, FLOAT_TYPE),
            'one_hot': pocket_one_hot.to(device, FLOAT_TYPE),
            'size': pocket_size.to(device, INT_TYPE),
            'mask': pocket_mask.to(device, INT_TYPE)
        }

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        if num_nodes_lig is None:
            num_nodes_lig = model.ddpm.size_distribution.sample_conditional(n1=None, n2=pocket['size'])

        # statistical data
        all_molecules = []
        valid_molecules = []
        processed_molecules = []
        n_generated = 0
        n_valid = 0
        iter=0

        while len(valid_molecules) < args.n_samples:
            iter += 1
            print('iter: ',iter)
            if type(model.ddpm) == ConditionalDDPM: 
                xh_lig, xh_pocket, lig_mask, pocket_mask = \
                    model.ddpm.sample_given_pocket(pocket, num_nodes_lig,
                                              timesteps=args.timesteps)
        
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
                                        sanitize=args.sanitize,
                                        relax_iter=0,
                                        largest_frag=False)
                if mol is not None:
                    mols_batch.append(mol)

            all_molecules.extend(mols_batch)
            # Filter to find valid molecules
            mols_batch_processed = [
                process_molecule(m, sanitize=args.sanitize,
                                relax_iter=(200 if args.relax else 0),
                                largest_frag=not args.all_frags)
                for m in mols_batch
            ]
            processed_molecules.extend(mols_batch_processed)
            valid_mols_batch = [m for m in mols_batch_processed if m is not None]
  
            n_generated += args.batch_size
            n_valid += len(valid_mols_batch)
            valid_molecules.extend(valid_mols_batch)
  
        valid_molecules = valid_molecules[:args.n_samples]

        # Reorder raw files
        all_molecules = \
            [all_molecules[i] for i, m in enumerate(processed_molecules)
            if m is not None] + \
            [all_molecules[i] for i, m in enumerate(processed_molecules)
            if m is None]
        
        # Write SDF files
        utils.write_sdf_file(sdf_out_file_raw, all_molecules)
        utils.write_sdf_file(sdf_out_file_processed, valid_molecules)
        
        # Time the sampling process
        time_per_pocket[str(sdf_file)] = time() - t_pocket_start
        with open(time_file, 'w') as f:
            f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

        pbar.set_description(
            f'Last processed: {sdf_file}. '
            f'n_valid: {n_valid} n_generated: {n_generated} '
            f'Validity: {n_valid / n_generated * 100:.2f}%. '
            f'{(time() - t_pocket_start) / len(valid_molecules):.2f} '
            f'sec/mol. '
            f'pocket: {pocket_size[0]} '
        )
 
    with open(Path(args.outdir, 'pocket_times.txt'), 'w') as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(f"Time per pocket: {times_arr.mean():.3f} \pm "
          f"{times_arr.std(unbiased=False):.2f}")
    
    print('total time: ', time()-start)
        
