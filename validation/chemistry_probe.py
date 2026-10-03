import argparse,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--sampling');p.add_argument('--out');a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
import torch,numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from analysis.molecule_builder import build_molecule,process_molecule
from analysis.SA_Score.sascorer import calculateScore
from constants import dataset_params
sample=np.load(a.sampling);z=sample['ligand'];mask=sample['mask'];r=[]
for i in np.unique(mask):
 x=z[mask==i];m=build_molecule(torch.from_numpy(x[:,:3]),torch.from_numpy(x[:,3:].argmax(1)),dataset_params['crossdock_h'],add_coords=True)
 graph=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType())) for b in m.GetBonds()]
 m=process_molecule(m,sanitize=True,relax_iter=200,largest_frag=False)
 r.append({'sample':int(i),'graph':graph,'valid':m is not None,'smiles':Chem.MolToSmiles(m) if m is not None else None,'qed':QED.qed(m) if m is not None else None,'sa':calculateScore(m) if m is not None else None})
Path(a.out).write_text(json.dumps(r,indent=2));print('CHEMISTRY_PROBE_DONE')
