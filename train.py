import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel

from datetime import datetime
import pytz

kst = pytz.timezone("Asia/Seoul")  # 한국 표준시 (KST) 설정
now_kst = datetime.now(kst)  # 현재 한국 시간 가져오기
formatted_time_kst = now_kst.strftime("%Y%m%d%H%M%S")  # YYYYMMDDHHMMSS 형태로 포맷팅


def get_argparse_groups(parser, args):
    """Group argparse arguments by their group title."""
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == "__main__":
    # Create parser
    parser = ArgumentParser()

    # Add model and data arguments
    parser.add_argument(
        "--backbone",
        type=str,
        choices=BackboneRegistry.get_all_names(),
        default="ncsnpp",
    )
    parser.add_argument(
        "--ode",
        type=str,
        choices=ODERegistry.get_all_names(),
        default="flowmatching",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Turn off logging to W&B, using local default logger instead",
    )

    # Trainer arguments (Lightning 2.x style)
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="Max training epochs"
    )
    parser.add_argument(
        "--devices", type=str, default="auto", help="Number of GPUs or 'auto'"
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=1,
        help="Number of sanity validation steps",
    )

    # Add VFModel arguments
    VFModel.add_argparse_args(
        parser.add_argument_group("VFModel", description=VFModel.__name__)
    )

    # Add ODE arguments
    temp_args, _ = parser.parse_known_args()
    ode_class = ODERegistry.get_by_name(temp_args.ode)
    ode_class.add_argparse_args(
        parser.add_argument_group("ODE", description=ode_class.__name__)
    )

    # Add backbone arguments
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__)
    )

    # Add data module arguments
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__)
    )

    # Parse all arguments
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)
    dataset = os.path.basename(os.path.normpath(args.base_dir))

    # Initialize model
    model = VFModel(
        backbone=args.backbone,
        ode=args.ode,
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups["VFModel"]),
            **vars(arg_groups["ODE"]),
            **vars(arg_groups["Backbone"]),
            **vars(arg_groups["DataModule"]),
        },
    )

    # Set up logger configuration
    name_save_dir_path = f"dataset_{dataset}_{formatted_time_kst}"

    if args.no_wandb:
        logger = TensorBoardLogger(save_dir="logs", name=name_save_dir_path)
    else:
        logger = WandbLogger(
            project="FLOWSE", log_model=True, save_dir="logs", name=name_save_dir_path
        )
        logger.experiment.log_code(".")

    # Set up callbacks
    model_dirpath = f"logs/{name_save_dir_path}"
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=model_dirpath, save_last=True, filename="{epoch}_last"
    )
    checkpoint_callback_pesq = ModelCheckpoint(
        dirpath=model_dirpath,
        save_top_k=20,
        monitor="pesq",
        mode="max",
        filename="{epoch}_{pesq:.2f}",
    )
    checkpoint_callback_si_sdr = ModelCheckpoint(
        dirpath=model_dirpath,
        save_top_k=20,
        monitor="si_sdr",
        mode="max",
        filename="{epoch}_{si_sdr:.2f}",
    )
    callbacks = [
        checkpoint_callback_last,
        checkpoint_callback_pesq,
        checkpoint_callback_si_sdr,
    ]

    # Parse devices argument
    if args.devices == "auto":
        devices = "auto"
    else:
        try:
            devices = int(args.devices)
        except ValueError:
            # Could be a list like [0,1,2,3]
            devices = args.devices

    # Initialize the Trainer (Lightning 2.x style)
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

    # Train model
    trainer.fit(model)
