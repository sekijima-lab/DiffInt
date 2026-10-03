"""Optional real training/resume and generation CLI smoke in a new output directory."""
import argparse,json,os,shutil,subprocess,sys
from argparse import Namespace
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
import numpy as np,torch,yaml
from diffint_runtime.checkpoints import load_checkpoint
out=args.out.resolve()
if out.exists() and any(out.iterdir()):raise ValueError('Choose a new empty output directory')
out.mkdir(parents=True,exist_ok=True)
for name in ['train','val']:shutil.copyfile(repo/'validation/legacy-input.npz',out/f'{name}.npz')
parameters=load_checkpoint(repo/'checkpoints/best_model.ckpt')['hyper_parameters']
np.save(out/'size_distribution.npy',np.array(parameters['node_histogram']))
config={k:vars(v) if isinstance(v,Namespace) else v for k,v in parameters.items() if k not in ['node_histogram','outdir']}
config.update(logdir=str(out),datadir=str(out),n_epochs=1,num_workers=0,batch_size=1,gpus=1,accelerator='cpu',enable_progress_bar=False,num_sanity_val_steps=0,eval_epochs=1000,visualize_sample_epoch=1000,visualize_chain_epoch=1000,wandb_params={'mode':'offline','entity':None})
env=dict(os.environ,OMP_NUM_THREADS='1',PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
results={}
for mode in ['compatible','standard']:
    config.update(run_name=mode,n_epochs=1)
    flags=[] if mode=='compatible' else ['--no-old-compatible']
    path=out/f'{mode}.yaml';path.write_text(yaml.safe_dump(config))
    for epoch in [1,2]:
        config['n_epochs']=epoch;path.write_text(yaml.safe_dump(config))
        command=[sys.executable,str(repo/'train.py'),'--config',str(path),*flags]
        checkpoint=out/mode/'checkpoints/last.ckpt'
        if epoch==2:command+=['--resume',str(checkpoint)]
        with (out/f'{mode}-epoch{epoch}.log').open('w') as log:
            subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    saved=max([load_checkpoint(p) for p in checkpoint.parent.glob('last*.ckpt')],key=lambda c:c['global_step'])
    assert saved['global_step']==2
    assert saved['loops']['fit_loop']['epoch_progress']['current']['processed']==2
    assert saved['hyper_parameters']['old_compatible']==(mode=='compatible')
    results[mode]={'global_step':2,'processed_epochs':2,'mode_saved':saved['hyper_parameters']['old_compatible'],'optimizer_states':len(saved['optimizer_states'][0]['state'])}
(out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
print('Both training modes and optimizer resume passed. Results:',out/'result.json')
