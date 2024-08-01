import oddt
import numpy as np
from oddt.interactions import hbonds, hbond_acceptor_donor
from tqdm import tqdm
import argparse
from pathlib import Path

def hbond_create(protein, ligand):
    li_do = hbond_acceptor_donor(protein,ligand)
    li_ac = hbond_acceptor_donor(ligand, protein)

    # ligand pocket interaction id
    po_id_d = [li_do[0][i]['id'] for i in range(len(li_do[0])) if li_do[2][i]==True]
    po_id_a = [li_ac[1][i]['id'] for i in range(len(li_ac[0])) if li_ac[2][i]==True]
 
    li_id_d = [li_do[1][i]['id'] for i in range(len(li_do[0])) if li_do[2][i]==True]
    li_id_a = [li_ac[0][i]['id'] for i in range(len(li_ac[0])) if li_ac[2][i]==True]

    int_id = []
    if len(li_id_d)>0:
        [int_id.append([li_id_d[i],0,po_id_d[i],0])for i in range(len(li_id_d))]
    if len(li_id_a)>0:
        [int_id.append([0,li_id_a[i],0,po_id_a[i]])for i in range(len(li_id_a))]
    
    # interaction particle coords
    coords_lido_1 = [(2*li_do[0][i]['coords']+li_do[1][i]['coords'])/3 for i in range(len(li_do[0])) if li_do[2][i]==True]
    coords_lido_2 = [(li_do[0][i]['coords']+2*li_do[1][i]['coords'])/3 for i in range(len(li_do[0])) if li_do[2][i]==True]
    coords_liac_1 = [(2*li_ac[0][i]['coords']+li_ac[1][i]['coords'])/3 for i in range(len(li_ac[0])) if li_ac[2][i]==True]
    coords_liac_2 = [(li_ac[0][i]['coords']+2*li_ac[1][i]['coords'])/3 for i in range(len(li_ac[0])) if li_ac[2][i]==True]

    inter_coords=[]
    [inter_coords.append(item) for pair in zip(coords_lido_1, coords_lido_2) for item in pair]
    [inter_coords.append(item) for pair in zip(coords_liac_1, coords_liac_2) for item in pair]
    inter_coords=np.array(inter_coords)

    len_do = len(coords_lido_1)*2
    len_ac = len(coords_liac_1)*2
    do_ac_id = np.concatenate([np.zeros(len_do,dtype=int),np.ones(len_ac,dtype=int)])
    inter_one_hot = np.identity(2)[do_ac_id]

    if len(int_id)==0:
        int_id.append([])

    return int_id, inter_coords, inter_one_hot

def data_create(file, base_path):
    inter_id = []
    inter_mask = []
    pocket_coords = []
    pocket_one_hot = []

    pdb_name = [x.split('.pdb_')[0]+'.pdb' for x in file['names']]
    lig_name = [x.split('.pdb_')[1] for x in file['names']]
    one_hot_len = len(file['pocket_one_hot'][0])

    po_one_tmp1 = np.append(file['pocket_one_hot'],np.zeros((len(file['pocket_one_hot']),2)),axis=1)
 
    sec_poc = np.where(np.diff(file['pocket_mask']))[0]+1
    po_cor_tmp2 = np.split(file['pocket_coords'],sec_poc)
    po_one_tmp2 = np.split(po_one_tmp1,sec_poc)

    for i in tqdm(range(len(file['names']))):
        protein = next(oddt.toolkit.readfile('pdb',str(Path(base_path, pdb_name[i]))))
        ligand = next(oddt.toolkit.readfile('sdf',str(Path(base_path, lig_name[i]))))
        protein.protein = True
        ligand.removeh()

        int_id, inter_coords, inter_one_hot = hbond_create(protein, ligand)
        if len(int_id[0])>0: [inter_id.append(x) for x in int_id]
        [inter_mask.append(x) for x in np.full(len(int_id),i)]

        # one hot
        if len(inter_one_hot)>0:
            po_int_cor_tmp3 = np.append(po_cor_tmp2[i],inter_coords,axis=0)
            inter_one_hot = np.append(np.zeros((len(inter_one_hot),one_hot_len)),inter_one_hot,axis=1)
            po_int_one_tmp3 = np.append(po_one_tmp2[i],inter_one_hot,axis=0)
        else:
            po_int_cor_tmp3 = po_cor_tmp2[i]
            po_int_one_tmp3 = po_one_tmp2[i]

        pocket_coords.append(po_int_cor_tmp3)
        pocket_one_hot.append(po_int_one_tmp3)

    ## mask
    po_mask = [x for i in range(len(file['names'])) for x in np.full(len(pocket_one_hot[i]),i)]
    po_mask = np.array(po_mask)

    inter_id = np.array(inter_id)
    inter_mask = np.array(inter_mask)

    pocket_coords = [y for x in pocket_coords for y in x]
    pocket_one_hot = [y for x in pocket_one_hot for y in x]

    pocket_coords = np.array(pocket_coords)
    pocket_one_hot = np.array(pocket_one_hot)

    return inter_id, inter_mask, pocket_coords, pocket_one_hot, po_mask

def data_save(data_dir, new_dir, pdb_dir, data_type):
    file_path = Path(data_dir,data_type + '.npz')
    data_file = np.load(file_path)

    print('=== data creation for ' + data_type + ' ===')
    inter_id, inter_mask, pocket_coords, pocket_one_hot, po_mask = data_create(data_file, pdb_dir)

    # data save
    out_name = Path(new_dir, data_type +'.npz')

    np.savez(out_name,
            names=data_file['names'],
            lig_coords=data_file['lig_coords'],
            lig_one_hot=data_file['lig_one_hot'],
            lig_mask=data_file['lig_mask'],
            pocket_coords=pocket_coords,
            pocket_one_hot=pocket_one_hot,
            pocket_mask=po_mask,
            inter_id=inter_id,
            inter_mask=inter_mask
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path,default=None)
    parser.add_argument('--out_dir', type=Path,default=None)
    parser.add_argument('--pdb_dir', type=Path,default=None)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=args.skip_existing)

    data_type = ['val', 'test', 'train']
    [data_save(args.data_dir, args.out_dir ,args.pdb_dir ,n) for n in data_type]

if __name__ == "__main__":
    main()