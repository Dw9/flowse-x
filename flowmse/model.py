import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec


class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)"
        )
        parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.999,
            help="The parameter EMA decay constant (0.999 by default)",
        )
        parser.add_argument(
            "--t_eps", type=float, default=0.03, help="t_delta in the paper"
        )
        parser.add_argument(
            "--T_rev", type=float, default=1.0, help="Starting point t_N in the paper"
        )

        parser.add_argument(
            "--num_eval_files",
            type=int,
            default=10,
            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).",
        )
        parser.add_argument(
            "--loss_type",
            type=str,
            default="mse",
            help="The type of loss function to use.",
        )
        parser.add_argument(
            "--loss_abs_exponent",
            type=float,
            default=0.5,
            help="magnitude transformation in the loss term",
        )

        return parser

    def __init__(
        self,
        backbone,
        ode,
        lr=1e-4,
        ema_decay=0.999,
        t_eps=0.03,
        T_rev=1.0,
        loss_abs_exponent=0.5,
        num_eval_files=10,
        loss_type="mse",
        data_module_cls=None,
        **kwargs,
    ):
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)

        ode_cls = ODERegistry.get_by_name(ode)
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=["no_wandb"])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get("gpus", 0) > 0)

        # Manual EMA for dnn parameters only (to avoid DDP compatibility issues)
        self.ema_dnn = {}
        for name, param in self.dnn.named_parameters():
            self.ema_dnn[name] = param.data.clone()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # Update EMA for dnn parameters only
        for name, param in self.dnn.named_parameters():
            if name in self.ema_dnn:
                # Move ema_dnn to same device as param if needed
                ema = self.ema_dnn[name]
                if ema.device != param.device:
                    ema = ema.to(param.device)
                    self.ema_dnn[name] = ema
                self.ema_dnn[name] = (
                    self.ema_decay * ema + (1 - self.ema_decay) * param.data
                )

    def on_load_checkpoint(self, checkpoint):
        ema_dnn = checkpoint.get("ema_dnn", None)
        if ema_dnn is not None:
            self.ema_dnn = ema_dnn
        else:
            ema_legacy = checkpoint.get("ema", None)
            if ema_legacy is not None and "shadow_params" in ema_legacy:
                shadow_params = ema_legacy["shadow_params"]
                # Legacy torch_ema tracks model.parameters() which may include
                # buffers registered as Parameters in the original code.
                # Map shadow_params via state_dict keys for correct alignment.
                sd_keys = list(checkpoint["state_dict"].keys())
                if len(shadow_params) == len(sd_keys):
                    for key, sp in zip(sd_keys, shadow_params):
                        # Strip 'dnn.' prefix for ema_dnn key
                        if key.startswith("dnn."):
                            dnn_key = key[4:]
                            # Only store if it's an actual parameter (not buffer)
                            if dnn_key in self.ema_dnn:
                                self.ema_dnn[dnn_key] = sp.cpu()
                    print("Migrated EMA from legacy torch_ema format (state_dict aligned)")
                else:
                    # Fallback: direct zip (same count)
                    for (name, _), sp in zip(self.dnn.named_parameters(), shadow_params):
                        self.ema_dnn[name] = sp.cpu()
                    print("Migrated EMA from legacy torch_ema format (direct zip)")
            else:
                self._error_loading_ema = True
                warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_dnn"] = self.ema_dnn

    def on_train_batch_end(self, *args, **kwargs):
        pass  # EMA update is handled in optimizer_step

    def on_validation_start(self):
        self._swap_to_ema()

    def on_validation_end(self):
        self._restore_from_ema()

    def on_test_start(self):
        self._swap_to_ema()

    def on_test_end(self):
        self._restore_from_ema()

    def _swap_to_ema(self):
        if self._error_loading_ema:
            return
        # Store current params and swap to EMA
        self._current_params = {}
        for name, param in self.dnn.named_parameters():
            if name in self.ema_dnn:
                self._current_params[name] = param.data.clone()
                param.data = self.ema_dnn[name].to(param.device)

    def _restore_from_ema(self):
        if self._error_loading_ema or not hasattr(self, "_current_params"):
            return
        # Restore original params
        for name, param in self.dnn.named_parameters():
            if name in self._current_params:
                param.data = self._current_params[name]
        self._current_params = {}

    def _mse_loss(self, x, x_hat):
        err = x - x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _loss(self, vectorfield, condVF):
        if self.loss_type == "mse":
            err = vectorfield - condVF
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            err = vectorfield - condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x0, y = batch
        B = x0.shape[0]
        rdm = (1 - torch.rand(B, device=x0.device)) * (
            self.T_rev - self.t_eps
        ) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.T_rev, device=x0.device))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)  #
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0, t, y)
        condVF = der_std * z + der_mean

        vectorfield = self(xt, t, y)

        loss = self._loss(vectorfield, condVF)

        return loss

    # ========== Training / Validation Steps ==========

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log("pesq", pesq, on_step=False, on_epoch=True)
            self.log("si_sdr", si_sdr, on_step=False, on_epoch=True)
            self.log("estoi", estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
