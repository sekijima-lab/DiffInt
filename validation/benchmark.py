import argparse,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--legacy',action='store_true');p.add_argument('--sample',action='store_true');a=p.parse_args()
repo=Path(a.repo).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=True,parents=True)
if a.legacy:import legacy_bootstrap
sys.path.insert(0,str(repo))
import numpy as np,torch
from lightning_modules import LigandPocketDDPM
from process_crossdock import process_ligand_and_pocket
from constants import dataset_params
from hbond_double import hbond_create
import oddt
from dataset import ProcessedLigandPocketDataset

torch.set_num_threads(1);torch.manual_seed(1729)
m=LigandPocketDDPM.load_from_checkpoint(str(repo/'checkpoints/best_model.ckpt'),map_location='cpu')
meta=dataset_params['crossdock']
lig,po=process_ligand_and_pocket(repo/'example/1a2g_A_rec.pdb',repo/'example/1a2g_A_rec.sdf',meta['atom_encoder'],8.0,meta['aa_encoder'],True)
protein=next(oddt.toolkit.readfile('pdb',str(repo/'example/1a2g_A_rec.pdb')));protein.protein=True
ligand=next(oddt.toolkit.readfile('sdf',str(repo/'example/1a2g_A_rec.sdf')));ligand.removeh()
ids,coords,hot=hbond_create(protein,ligand)
ph=np.pad(po['pocket_one_hot'],((0,0),(0,2)))
if len(hot):
 pc=np.concatenate([po['pocket_coords'],coords]);ph=np.concatenate([ph,np.pad(hot,((0,0),(20,0)))])
else:pc=po['pocket_coords']
inputs=dict(names=np.array(['1a2g_A_rec']),**lig,lig_mask=np.zeros(len(lig['lig_coords'])),pocket_coords=pc,pocket_one_hot=ph,pocket_mask=np.zeros(len(pc)),inter_id=np.array(ids,dtype=np.int64),inter_mask=np.zeros(len(ids)))
np.savez(out/'input.npz',**inputs)
ds=ProcessedLigandPocketDataset(out/'input.npz');data=ds.collate_fn([ds[0] for _ in range(4)])
lg,pk=m.get_ligand_and_pocket(data)
rng=np.random.RandomState(73)
xh=torch.from_numpy(rng.normal(size=(len(lg['x']),3+m.atom_nf)).astype('float32'))
xp=torch.cat([pk['x'],pk['one_hot']],-1)
results={}
for i,t in enumerate([0.,0.1,0.5,0.9,1.]):
 with torch.no_grad():
  pred=m.ddpm.dynamics(xh,xp,torch.tensor([[t]]),lg['mask'],pk['mask'])
 for j,v in enumerate(pred):results[f'forward_{i}_{j}']=v.numpy()
m.train()
for seed in [1729,19,73]:
 torch.manual_seed(seed);m.zero_grad();nll,info=m(data);loss=nll.mean();loss.backward()
 results[f'loss_{seed}']=loss.detach().numpy()
 results[f'grad_{seed}']=torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).numpy()
 if seed==73:
  opt=m.configure_optimizers();opt.step();results['weights_step']=torch.cat([p.detach().flatten() for p in m.parameters()]).numpy()
np.savez_compressed(out/'numeric.npz',**results)
print('NUMERIC_DONE',len(lg['x']),len(pk['x']),flush=True)
if a.sample:
 m=LigandPocketDDPM.load_from_checkpoint(str(repo/'checkpoints/best_model.ckpt'),map_location='cpu');m.eval()
 data=ds.collate_fn([ds[0] for _ in range(3)]);_,pk=m.get_ligand_and_pocket(data)
 torch.manual_seed(1729)
 with torch.no_grad():z,pocket,mask,_=m.ddpm.sample_given_pocket(pk,torch.tensor([12,16,20]))
 np.savez_compressed(out/'sampling.npz',ligand=z.numpy(),pocket=pocket.numpy(),mask=mask.numpy())
 print('SAMPLING_DONE',flush=True)
