import argparse
from argparse import Namespace
from pathlib import Path, PosixPath, WindowsPath
import warnings
import json
import sys

import torch
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM
from diffint_runtime.checkpoints import load_checkpoint


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--old-compatible', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    requested_math_mode = args.old_compatible

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = load_checkpoint(ckpt_path)['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)
    args.old_compatible = requested_math_mode if requested_math_mode is not None else config.get("old_compatible", True)
    args.accelerator = getattr(args, "accelerator", "cpu" if args.old_compatible else "gpu")

    if args.old_compatible and args.accelerator != "cpu":
        raise ValueError("Old-compatible mode requires accelerator: cpu; use --no-old-compatible for standard GPU math")

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()

    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        alpha_param=args.alpha_param,
        alpha_power=args.alpha_power,
        old_compatible=args.old_compatible
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime.json").write_text(json.dumps({
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "lightning": pl.__version__, "old_compatible": args.old_compatible,
        "accelerator": args.accelerator}, indent=2) + "\n")

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='DiffInt_training',
        group=args.dataset,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator=args.accelerator,
        devices=args.gpus if args.accelerator == 'gpu' else 1,
        strategy='ddp' if args.accelerator == 'gpu' and args.gpus > 1 else 'auto'
    )

    with torch.serialization.safe_globals([Namespace, Path, PosixPath, WindowsPath]):
        trainer.fit(model=pl_module, ckpt_path=ckpt_path, weights_only=True)
