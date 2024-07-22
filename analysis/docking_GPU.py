import os
import re
import tempfile
import numpy as np
import torch
from pathlib import Path
import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import time

try:
    import utils
except ModuleNotFoundError as e:
    print(e)


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def smina_score(rdmols, receptor_file):
    """
    Calculate smina score
    :param rdmols: List of RDKit molecules
    :param receptor_file: Receptor pdb/pdbqt file or list of receptor files
    :return: Smina score for each input molecule (list)
    """

    if isinstance(receptor_file, list):
        scores = []
        for mol, rec_file in zip(rdmols, receptor_file):
            with tempfile.NamedTemporaryFile(suffix='.sdf') as tmp:
                tmp_file = tmp.name
                utils.write_sdf_file(tmp_file, [mol])
                scores.extend(calculate_smina_score(rec_file, tmp_file))

    # Use same receptor file for all molecules
    else:
        with tempfile.NamedTemporaryFile(suffix='.sdf') as tmp:
            tmp_file = tmp.name
            utils.write_sdf_file(tmp_file, rdmols)
            scores = calculate_smina_score(receptor_file, tmp_file)

    return scores


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} '
             f'-f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False):

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        os.popen(f'prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    aa, bb, cc = [],[],[]
    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        ligand_name = f'{sdf_file.stem}_{i}'
        ligand_pdbqt_dir = Path(out_dir, ligand_name.split('_')[0])
        if not os.path.exists(ligand_pdbqt_dir): os.mkdir(ligand_pdbqt_dir)
        ligand_pdbqt_file = Path(ligand_pdbqt_dir, ligand_name + '.pdbqt')
        # prepare ligand
        #ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        #out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)
        aa.append(cx)
        bb.append(cy)
        cc.append(cz)

    cx = np.average(aa)
    cy = np.average(bb)
    cz = np.average(cc)
    start = time.time()
    print('== docking start ==')
    # docking GPU
    print(receptor_pdbqt_file)
    print(ligand_pdbqt_dir)
    out = os.popen(
    f'Vina-GPU --receptor {receptor_pdbqt_file} '
    f'--ligand_directory {ligand_pdbqt_dir} '
    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
    f'--size_x {size} --size_y {size} --size_z {size} --thread 8000 '
    f'--search_depth 16'
    ).read()
    print('== docking finish ==')
    print('docking time = ', time.time()-start)
    out_split = out.splitlines()
    best_idxs = [i+1 for i, x in enumerate(out_split) if x == '-----+------------+----------+----------']
    scores = [float(out_split[i].split()[1]) for i in best_idxs]

    for i in range(len(suppl)):
        ligand_name = f'{sdf_file.stem}_{i}'
        out_pdbqt_dir = Path(out_dir, ligand_name.split('_')[0]+'_out')
        out_pdbqt_file = Path(out_pdbqt_dir, ligand_name +'_out.pdbqt')
        out_sdf_file = Path(out_pdbqt_dir, ligand_name +'_out.sdf')
        if out_pdbqt_file.exists():
            os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()
    
        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='moad')
    args = parser.parse_args()
    program_start = time.time()
    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    args.out_dir.mkdir(exist_ok=True)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            #receptor_name = ligand_name[:-4]
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            print(sdf_file)
            #receptor_name = sdf_file.stem.split('.pdb')[0]
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
            print(receptor_file)

        # try:
        scores, rdmols = calculate_qvina2_score(
            receptor_file, sdf_file, args.out_dir, return_rdmol=True)
        # except AttributeError as e:
        #     print(e)
        #     continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[ligand_name] = {
                'receptor': str(receptor_file),
                'ligand': str(sdf_file),
                'scores': scores,
                'rmdols': rdmols
            }

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))

    print('total time = ', time.time()-program_start)
