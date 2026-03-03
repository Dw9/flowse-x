"""
Training script for DriftingSE — Drifting Model for Speech Enhancement.

Usage:
  python train_drifting.py --base_dir <dataset_dir> [--loss_type hybrid] [--no_wandb]

Examples:
  # Hybrid loss (drifting + MSE reconstruction) — recommended
  python train_drifting.py --base_dir /path/to/voicebank --loss_type hybrid --no_wandb

  # Pure drifting loss (experimental)
  python train_drifting.py --base_dir /path/to/voicebank --loss_type drifting --no_wandb

  # Pure MSE baseline (for comparison)
  python train_drifting.py --base_dir /path/to/voicebank --loss_type mse --no_wandb
"""

import argparse
from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.drifting_model import DriftingSEModel

from datetime import datetime
import pytz

kst = pytz.timezone("Asia/Seoul")
now_kst = datetime.now(kst)
formatted_time_kst = now_kst.strftime("%Y%m%d%H%M%S")


def get_argparse_groups(parser, args):
    """Group argparse arguments by their group title."""
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model selection
    parser.add_argument(
        "--backbone",
        type=str,
        choices=BackboneRegistry.get_all_names(),
        default="ncsnpp",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Use TensorBoard instead of W&B",
    )

    # Trainer arguments
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--num_sanity_val_steps", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint for weight initialization")

    # DriftingSEModel arguments
    DriftingSEModel.add_argparse_args(
        parser.add_argument_group("DriftingSEModel", description="DriftingSEModel")
    )

    # Backbone arguments
    temp_args, _ = parser.parse_known_args()
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__)
    )

    # Data module arguments
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__)
    )

    # Parse
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)
    dataset = os.path.basename(os.path.normpath(args.base_dir))

    # Initialize model
    model = DriftingSEModel(
        backbone=args.backbone,
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups["DriftingSEModel"]),
            **vars(arg_groups["Backbone"]),
            **vars(arg_groups["DataModule"]),
        },
    )

    # Logger
    name_save_dir_path = f"drifting_{dataset}_{args.loss_type}_{formatted_time_kst}"

    if args.no_wandb:
        logger = TensorBoardLogger(save_dir="logs", name=name_save_dir_path)
    else:
        logger = WandbLogger(
            project="DRIFTING-SE", log_model=True, save_dir="logs",
            name=name_save_dir_path,
        )
        logger.experiment.log_code(".")

    # Callbacks
    model_dirpath = f"logs/{name_save_dir_path}"
    callbacks = [
        ModelCheckpoint(dirpath=model_dirpath, save_last=True, filename="{epoch}_last"),
        ModelCheckpoint(
            dirpath=model_dirpath, save_top_k=20, monitor="pesq",
            mode="max", filename="{epoch}_{pesq:.2f}",
        ),
        ModelCheckpoint(
            dirpath=model_dirpath, save_top_k=20, monitor="si_sdr",
            mode="max", filename="{epoch}_{si_sdr:.2f}",
        ),
    ]

    # Devices
    if args.devices == "auto":
        devices = "auto"
    else:
        try:
            devices = int(args.devices)
        except ValueError:
            devices = args.devices

    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy="auto",
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
    )

    # Load pretrained weights if specified
    if args.ckpt:
        import torch as _torch
        print(f"Loading pretrained weights from: {args.ckpt}")
        ckpt = _torch.load(args.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        model.on_load_checkpoint(ckpt)
        print("  Pretrained weights loaded successfully!")

    # Train
    trainer.fit(model)
