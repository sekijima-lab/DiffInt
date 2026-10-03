import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from argparse import Namespace
from diffint_runtime.scatter import scatter_add,scatter_mean
from diffint_runtime.checkpoints import load_checkpoint
from diffint_runtime.residues import three_to_one
from dataset import ProcessedLigandPocketDataset
class UnsupportedObject:
    def __reduce__(self):return (eval,('42',))
class RuntimeTests(unittest.TestCase):
    def test_batch_reductions_and_gradients(self):
        indices=torch.tensor([0,2,0,2]);x=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],requires_grad=True)
        total=scatter_add(x,indices,dim_size=4);mean=scatter_mean(x,indices,dim_size=4)
        torch.testing.assert_close(total,torch.tensor([[6.,8.],[0.,0.],[10.,12.],[0.,0.]]),rtol=0,atol=0)
        torch.testing.assert_close(mean,total/torch.tensor([[2.],[1.],[2.],[1.]]),rtol=0,atol=0)
        mean.sum().backward();torch.testing.assert_close(x.grad,torch.full_like(x,0.5),rtol=0,atol=0)
        self.assertEqual(scatter_add(torch.empty((0,3)),torch.empty(0,dtype=torch.long)).shape,(0,3))
        with self.assertRaises(ValueError):scatter_add(x,indices,dim=1)
    def test_checkpoint_allowlist_and_pickle_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.ckpt';obj={'state_dict':{'w':torch.arange(4.)},'hyper_parameters':{'config':Namespace(a=1)}}
            torch.save(obj,p);loaded=load_checkpoint(p);self.assertEqual(loaded['hyper_parameters']['config'].a,1)
            torch.testing.assert_close(obj['state_dict']['w'],loaded['state_dict']['w'],rtol=0,atol=0)
            torch.save({'state_dict':{},'bad':UnsupportedObject()},p)
            with self.assertRaises(Exception):load_checkpoint(p)
            torch.save({'state_dict':{'bad':'non-tensor'}},p)
            with self.assertRaises(ValueError):load_checkpoint(p)
    def test_numpy_object_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unsafe.npz';np.savez(p,names=np.array([UnsupportedObject()],dtype=object))
            with self.assertRaises(ValueError):ProcessedLigandPocketDataset(p)
    def test_old_activations_and_first_gradients(self):
        from diffint_runtime.activations import _Activation
        baseline=np.load(Path(__file__).resolve().parents[1]/'validation/legacy-activations.npz',allow_pickle=False)
        for name in ['silu','sigmoid','tanh']:
            x=torch.from_numpy(baseline['x'].copy()).requires_grad_()
            y=_Activation.apply(x,name);y.sum().backward()
            torch.testing.assert_close(y,torch.from_numpy(baseline[name]),rtol=0,atol=0)
            torch.testing.assert_close(x.grad,torch.from_numpy(baseline[name+'_gradient']),rtol=0,atol=1e-7)
        with self.assertRaises(ValueError):_Activation.apply(torch.ones(3,dtype=torch.float64),'silu')
        from diffint_runtime import _legacy_math
        with self.assertRaises(ValueError):_legacy_math.exp(np.ones(3,dtype='float64'),np.empty(3,dtype='float64'))
        with self.assertRaises(ValueError):_legacy_math.exp(np.ones(3,dtype='float32'),np.empty(2,dtype='float32'))

    def test_model_mode_save_reload_and_override(self):
        from lightning_modules import LigandPocketDDPM
        from diffint_runtime.activations import CompatibleSiLU
        torch.set_num_threads(1)
        original=Path(__file__).resolve().parents[1]/'checkpoints/best_model.ckpt'
        with tempfile.TemporaryDirectory() as d:
            for enabled in [True,False]:
                m=LigandPocketDDPM.load_from_checkpoint(original,old_compatible=enabled)
                p=Path(d)/'model.ckpt'
                torch.save({'state_dict':m.state_dict(),'hyper_parameters':dict(m.hparams)},p)
                loaded=LigandPocketDDPM.load_from_checkpoint(p)
                self.assertEqual(loaded.old_compatible,enabled)
                self.assertEqual(any(isinstance(v,CompatibleSiLU) for v in loaded.modules()),enabled)
                switched=LigandPocketDDPM.load_from_checkpoint(p,old_compatible=not enabled)
                self.assertEqual(switched.old_compatible,not enabled)
                self.assertTrue(all(torch.equal(v,switched.state_dict()[k]) for k,v in m.state_dict().items()))

    def test_legacy_amino_acid_mapping(self):
        mapping=dict(zip(['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR'],'ACDEFGHIKLMNPQRSTVWY'))
        for key,value in mapping.items():self.assertEqual(three_to_one(key),value)
        for key in ['MSE','UNK','ala']:
            with self.assertRaises(KeyError):three_to_one(key)
if __name__=='__main__':unittest.main()
