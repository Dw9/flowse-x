import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec
from flowmse.losses import si_sdr_loss, estimate_x0_from_velocity
import numpy as np
import matplotlib.pyplot as plt
import random


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

        # PESQ + SI-SDR auxiliary loss arguments (from paper https://arxiv.org/pdf/2512.10382v1)
        parser.add_argument(
            "--use_pesq_loss",
            action="store_true",
            help="Enable PESQ auxiliary loss for better perceptual quality",
        )
        parser.add_argument(
            "--use_si_sdr_loss",
            action="store_true",
            help="Enable SI-SDR auxiliary loss",
        )
        parser.add_argument(
            "--alpha_pesq",
            type=float,
            default=5e-2,
            help="Weight for PESQ loss. Paper recommends: 5e-2 for velocity, 1e-3 for x1, 1e-6 for x1-EDM",
        )
        parser.add_argument(
            "--alpha_si_sdr",
            type=float,
            default=5e-3,
            help="Weight for SI-SDR loss. Paper recommends: 5e-3 for velocity, 1e-4 for x1, 1e-7 for x1-EDM",
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
        use_pesq_loss=False,
        use_si_sdr_loss=False,
        alpha_pesq=5e-2,
        alpha_si_sdr=5e-3,
        **kwargs,
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a vector field model.
            ode: The ode used.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
            use_pesq_loss: Whether to use PESQ auxiliary loss (default: False)
            use_si_sdr_loss: Whether to use SI-SDR auxiliary loss (default: False)
            alpha_pesq: Weight for PESQ loss (default: 5e-2, paper recommends based on objective type)
            alpha_si_sdr: Weight for SI-SDR loss (default: 5e-3, paper recommends based on objective type)
        """
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

        # PESQ + SI-SDR auxiliary loss settings (from paper https://arxiv.org/pdf/2512.10382v1)
        self.use_pesq_loss = use_pesq_loss
        self.use_si_sdr_loss = use_si_sdr_loss
        self.alpha_pesq = alpha_pesq
        self.alpha_si_sdr = alpha_si_sdr

        # Initialize PESQ module if needed
        self.pesq_module = None
        if self.use_pesq_loss:
            try:
                from torch_pesq import PesqLoss

                self.pesq_module = PesqLoss(factor=1.0, sample_rate=16000)
            except ImportError:
                print("Warning: torch-pesq not available, PESQ loss will be disabled")
                self.use_pesq_loss = False

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
                for name, param in zip(self.dnn.named_parameters(), shadow_params):
                    self.ema_dnn[name[0]] = param.cpu()
                print("Migrated EMA from legacy torch_ema format")
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
        rdm = (1 - torch.rand(x0.shape[0], device=x0.device)) * (
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

        # Compute PESQ + SI-SDR auxiliary losses if enabled
        # Based on paper: https://arxiv.org/pdf/2512.10382v1
        # L_total = L_CFM + α_p * L_PESQ + α_s * L_SI-SDR
        aux_loss_dict = {}

        if self.use_pesq_loss or self.use_si_sdr_loss:
            # Estimate x0 (clean) from velocity: x0_hat = xt - t * v_θ
            # NOTE: This codebase uses t=0→clean, t=1→noisy (opposite of paper 2512.10382)
            x1_hat_spec = estimate_x0_from_velocity(xt, vectorfield, t)

            # x1_hat_spec and x0 are complex tensors with shape (batch, 1, freq, time)
            # Squeeze channel dim to get (batch, freq, time) for to_audio
            x1_hat_audio = self.to_audio(x1_hat_spec.squeeze(1))
            x0_audio = self.to_audio(x0.squeeze(1))

            # Ensure 2D shape (batch, time)
            if x1_hat_audio.dim() == 3:
                x1_hat_audio = x1_hat_audio.squeeze(1)
            if x0_audio.dim() == 3:
                x0_audio = x0_audio.squeeze(1)

            # SI-SDR loss
            if self.use_si_sdr_loss:
                si_sdr_loss_val = si_sdr_loss(x1_hat_audio, x0_audio)
                loss = loss + self.alpha_si_sdr * si_sdr_loss_val
                aux_loss_dict["si_sdr_loss"] = si_sdr_loss_val

            # PESQ loss (PesqLoss.forward returns distance loss in [0, inf), lower is better)
            if self.use_pesq_loss and self.pesq_module is not None:
                try:
                    pesq_loss_val = self.pesq_module(x0_audio, x1_hat_audio).mean()
                    loss = loss + self.alpha_pesq * pesq_loss_val
                    aux_loss_dict["pesq_loss"] = pesq_loss_val
                except Exception as e:
                    warnings.warn(f"PESQ computation failed: {e}")

        return loss, aux_loss_dict

    def training_step(self, batch, batch_idx):
        loss, aux_loss_dict = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Log auxiliary losses
        for key, value in aux_loss_dict.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux_loss_dict = self._step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        # Log auxiliary losses
        for key, value in aux_loss_dict.items():
            self.log(f"valid_{key}", value, on_step=False, on_epoch=True)

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
