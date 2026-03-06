"""
Drifting Model for Speech Enhancement (DriftingSE).

Applies the Drifting Models paradigm (Deng et al., 2026, "Generative Modeling
via Drifting") to speech enhancement. Instead of multi-step ODE integration
(FlowSE), the model learns a single-step mapping from noisy to clean speech.

Key adaptations for SE:
  - Conditional generation: generator takes noisy spectrogram y as input
  - Residual learning: output = y + backbone(y, ε) for stable training
  - Hybrid loss: drifting (distribution matching) + MSE (per-sample fidelity)
  - Multi-scale features: spectrogram statistics + mel-spectrogram features
  - Energy-adaptive conditioning: input log-energy replaces fixed t=1.0
  - Backbone reuse: NCSNpp with FiLM conditioning driven by input energy
"""

import math
import warnings
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from flowmse.backbones import BackboneRegistry
from flowmse.drifting import compute_drifting_loss, create_mel_filterbank, FeatureBank
from flowmse.util.other import si_sdr, pad_spec

# Lazy import for evaluation
_pesq = None
_stoi = None


def _import_metrics():
    global _pesq, _stoi
    if _pesq is None:
        from pesq import pesq
        from pystoi import stoi
        _pesq = pesq
        _stoi = stoi


class DriftingSEModel(pl.LightningModule):
    """
    Single-step speech enhancement via drifting field training.

    The generator maps noisy speech directly to clean speech in one forward pass.
    Training uses a hybrid loss:
      L = λ_drift * L_drifting + λ_recon * L_reconstruction
    where L_drifting matches output distribution to clean speech distribution,
    and L_reconstruction ensures per-utterance fidelity.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999,
                            help="EMA decay for backbone parameters")
        parser.add_argument("--num_eval_files", type=int, default=150,
                            help="Number of validation files for evaluation")
        parser.add_argument("--loss_type", type=str, default="hybrid",
                            choices=["drifting", "hybrid", "mse"],
                            help="Loss type: 'drifting', 'hybrid' (drift+mse), or 'mse' (baseline)")
        parser.add_argument("--recon_weight", type=float, default=1.0,
                            help="Weight for reconstruction loss in hybrid mode")
        parser.add_argument("--drift_weight", type=float, default=1.0,
                            help="Weight for drifting loss")
        parser.add_argument("--temperatures", type=str, default="0.02,0.05,0.2",
                            help="Comma-separated temperature values for drifting kernel")
        parser.add_argument("--use_residual", type=int, default=1,
                            help="Use residual connection: x_hat = y + backbone(y,ε)")
        parser.add_argument("--stochastic", type=int, default=0,
                            help="Use stochastic noise input (1) or deterministic zeros (0)")
        parser.add_argument("--use_mel_features", type=int, default=1,
                            help="Use mel-spectrogram features in drifting loss (1) or not (0)")
        parser.add_argument("--n_mels", type=int, default=80,
                            help="Number of mel bands for mel features")
        parser.add_argument("--energy_conditioning", type=int, default=1,
                            help="Use input energy as conditioning (1) or fixed t=1.0 (0)")
        parser.add_argument("--bank_size", type=int, default=256,
                            help="Feature memory bank size for positive samples (0=disabled)")
        parser.add_argument("--complex_loss_weight", type=float, default=0.3,
                            help="Weight of complex MSE relative to log-magnitude MSE in recon loss")
        # LR scheduler
        parser.add_argument("--lr_scheduler", type=str, default="cosine",
                            choices=["none", "cosine"],
                            help="LR scheduler: 'none' (flat) or 'cosine' (warmup + cosine decay)")
        parser.add_argument("--warmup_epochs", type=int, default=5,
                            help="Linear warmup epochs before cosine decay")
        parser.add_argument("--lr_t_max", type=int, default=300,
                            help="Total epoch budget for cosine schedule (including warmup)")
        parser.add_argument("--lr_min", type=float, default=1e-6,
                            help="Minimum learning rate at end of cosine decay")
        # Multi-resolution STFT loss
        parser.add_argument("--use_mr_stft", type=int, default=1,
                            help="Use multi-resolution STFT loss (1=yes, 0=no)")
        parser.add_argument("--mr_stft_weight", type=float, default=0.01,
                            help="Weight for multi-resolution STFT loss")
        # Compatibility with evaluate flow (t_eps, T_rev unused but kept for interface)
        parser.add_argument("--t_eps", type=float, default=0.03)
        parser.add_argument("--T_rev", type=float, default=1.0)
        parser.add_argument("--loss_abs_exponent", type=float, default=0.5,
                            help="Magnitude transform exponent (unused, for compat)")
        return parser

    def __init__(
        self,
        backbone,
        lr=1e-4,
        ema_decay=0.999,
        num_eval_files=150,
        loss_type="hybrid",
        recon_weight=1.0,
        drift_weight=1.0,
        temperatures="0.02,0.05,0.2",
        use_residual=1,
        stochastic=0,
        use_mel_features=1,
        n_mels=80,
        energy_conditioning=1,
        bank_size=256,
        complex_loss_weight=0.3,
        lr_scheduler="cosine",
        warmup_epochs=5,
        lr_t_max=300,
        lr_min=1e-6,
        use_mr_stft=1,
        mr_stft_weight=0.01,
        t_eps=0.03,
        T_rev=1.0,
        loss_abs_exponent=0.5,
        data_module_cls=None,
        **kwargs,
    ):
        super().__init__()

        # Initialize backbone DNN (same architecture as FlowSE)
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)

        # Hyperparameters
        self.lr = lr
        self.ema_decay = ema_decay
        self.num_eval_files = num_eval_files
        self.loss_type = loss_type
        self.recon_weight = recon_weight
        self.drift_weight = drift_weight
        self.temperatures = tuple(float(t) for t in temperatures.split(","))
        self.use_residual = bool(use_residual)
        self.stochastic = bool(stochastic)
        self.use_mel_features = bool(use_mel_features)
        self.energy_conditioning = bool(energy_conditioning)
        self.complex_loss_weight = complex_loss_weight
        self.lr_scheduler_type = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.lr_t_max = lr_t_max
        self.lr_min = lr_min
        self.use_mr_stft = bool(use_mr_stft)
        self.mr_stft_weight = mr_stft_weight
        self.t_eps = t_eps
        self.T_rev = T_rev
        self._error_loading_ema = False

        self.save_hyperparameters(ignore=["no_wandb"])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get("gpus", 0) > 0)

        # Mel filterbank for perceptual features in drifting loss
        if self.use_mel_features:
            # n_fft=510 → n_freqs=256, sr=16000 (FlowSE defaults)
            mel_fb = create_mel_filterbank(n_freqs=256, sr=16000, n_mels=n_mels)
            self.register_buffer("mel_fb", mel_fb)
        else:
            self.mel_fb = None

        # Manual EMA for DNN parameters
        self.ema_dnn = {}
        for name, param in self.dnn.named_parameters():
            self.ema_dnn[name] = param.data.clone()

        # Feature memory bank for positive samples (clean features)
        self.bank_size = bank_size
        if bank_size > 0 and loss_type != "mse":
            self.feature_bank = FeatureBank(max_size=bank_size)
        else:
            self.feature_bank = None

        # Multi-resolution STFT loss (waveform-domain supervision)
        if self.use_mr_stft:
            from flowmse.losses import MultiResolutionSTFTLoss
            self.mr_stft_loss_fn = MultiResolutionSTFTLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dnn.parameters(), lr=self.lr)
        if self.lr_scheduler_type == "none":
            return optimizer

        warmup = self.warmup_epochs
        t_max = self.lr_t_max
        eta_min_ratio = self.lr_min / self.lr

        def lr_lambda(epoch):
            if epoch < warmup:
                # Linear warmup from 1% to 100%
                return max(0.01, epoch / max(1, warmup))
            # Cosine decay to lr_min
            progress = (epoch - warmup) / max(1, t_max - warmup)
            progress = min(progress, 1.0)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # Update EMA for DNN parameters
        for name, param in self.dnn.named_parameters():
            if name in self.ema_dnn:
                ema = self.ema_dnn[name]
                if ema.device != param.device:
                    ema = ema.to(param.device)
                    self.ema_dnn[name] = ema
                self.ema_dnn[name] = (
                    self.ema_decay * ema + (1 - self.ema_decay) * param.data
                )

    # ========== EMA Checkpoint Methods ==========

    def on_load_checkpoint(self, checkpoint):
        ema_dnn = checkpoint.get("ema_dnn", None)
        if ema_dnn is not None:
            self.ema_dnn = ema_dnn
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_dnn"] = self.ema_dnn

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
        self._current_params = {}
        for name, param in self.dnn.named_parameters():
            if name in self.ema_dnn:
                self._current_params[name] = param.data.clone()
                param.data = self.ema_dnn[name].to(param.device)

    def _restore_from_ema(self):
        if self._error_loading_ema or not hasattr(self, "_current_params"):
            return
        for name, param in self.dnn.named_parameters():
            if name in self._current_params:
                param.data = self._current_params[name]
        self._current_params = {}

    # ========== Forward Pass ==========

    def forward(self, y, epsilon=None):
        """
        Single-step enhancement: noisy → clean.

        Args:
            y: [B, 1, F, T] noisy spectrogram (complex, in transformed domain)
            epsilon: [B, 1, F, T] optional noise (complex) for stochastic generation

        Returns:
            x_hat: [B, 1, F, T] estimated clean spectrogram (complex)
        """
        if epsilon is None:
            if self.stochastic:
                # Complex Gaussian noise
                epsilon = torch.randn_like(y)
            else:
                epsilon = torch.zeros_like(y)

        # Concatenate [epsilon, y] as 2-channel complex input
        # NCSNpp will decompose this to 4 real channels internally
        dnn_input = torch.cat([epsilon, y], dim=1)  # [B, 2, F, T] complex

        if self.energy_conditioning:
            # Energy-adaptive conditioning: feed log-energy of input to NCSNpp.
            # NCSNpp computes log(time_cond) → Fourier features → FiLM bias.
            # With energy conditioning: different noise levels → different FiLM bias
            # → backbone adapts processing to input noise level.
            mag = y.abs()  # [B, 1, F, T]
            log_energy = torch.log(mag.mean(dim=(1, 2, 3)) + 1e-8)  # [B]
            # Map to (0, 1) range via sigmoid; NCSNpp does log() internally,
            # so this gives log(sigmoid(x)) ∈ (-∞, 0), valid for Fourier embedding
            t = torch.sigmoid(log_energy)  # [B]
            # Clamp away from exact 0 to avoid log(0) in NCSNpp
            t = t.clamp(min=1e-5, max=1.0 - 1e-5)
        else:
            # Fixed t=1.0: time embedding becomes a learned constant bias
            t = torch.ones(y.shape[0], device=y.device)

        # Forward through backbone (no negation — we want direct output, not score)
        output = self.dnn(dnn_input, t)  # [B, 1, F, T] complex

        if self.use_residual:
            # Residual learning: at initialization backbone outputs ~0,
            # so x_hat ≈ y (reasonable starting point for SE)
            x_hat = y + output
        else:
            x_hat = output

        return x_hat

    # ========== Loss Functions ==========

    def _spec_to_real(self, spec):
        """Convert complex spectrogram to 2-channel real tensor for drifting loss."""
        # spec: [B, 1, F, T] complex → [B, 2, F, T] real
        return torch.cat([spec.real, spec.imag], dim=1)

    def _reconstruction_loss(self, x_hat, x_clean):
        """Hybrid log-magnitude + complex MSE loss.

        Log-magnitude MSE: correlates well with PESQ (spectral envelope).
        Complex MSE (configurable weight): preserves waveform fidelity (SI-SDR).
        """
        mag_hat = x_hat.abs()
        mag_clean = x_clean.abs()
        log_mag_loss = F.mse_loss(torch.log(mag_hat + 1e-8), torch.log(mag_clean + 1e-8))
        complex_loss = F.mse_loss(x_hat.real, x_clean.real) + F.mse_loss(x_hat.imag, x_clean.imag)
        loss = log_mag_loss + self.complex_loss_weight * complex_loss
        self.log("log_mag_loss", log_mag_loss, on_step=True, on_epoch=True)
        self.log("complex_loss", complex_loss, on_step=True, on_epoch=True)
        return loss

    def _step(self, batch, batch_idx):
        x0, y = batch  # x0: clean spec (complex), y: noisy spec (complex)

        # Single-step generation
        x_hat = self.forward(y)

        # Multi-resolution STFT loss (waveform domain)
        mr_loss = None
        if self.use_mr_stft:
            wav_length = (x0.shape[-1] - 1) * self.data_module.hop_length
            x_hat_wav = self._istft(
                self._backward_transform(x_hat).squeeze(1), wav_length
            )
            with torch.no_grad():
                x0_wav = self._istft(
                    self._backward_transform(x0).squeeze(1), wav_length
                )
            mr_loss = self.mr_stft_loss_fn(x_hat_wav, x0_wav)
            self.log("mr_stft_loss", mr_loss, on_step=True, on_epoch=True)

        if self.loss_type == "mse":
            # Pure MSE baseline (no drifting)
            loss = self._reconstruction_loss(x_hat, x0)
            self.log("recon_loss", loss, on_step=True, on_epoch=True)
            if mr_loss is not None:
                loss = loss + self.mr_stft_weight * mr_loss
            return loss

        # Convert to real tensors for drifting loss computation
        x_hat_real = self._spec_to_real(x_hat)
        x_clean_real = self._spec_to_real(x0)

        # Drifting loss (distribution matching)
        loss_drift, v_mag = compute_drifting_loss(
            x_hat_real, x_clean_real.detach(),
            temperatures=self.temperatures,
            mel_fb=self.mel_fb,
            feature_bank=self.feature_bank if self.training else None,
        )

        if self.loss_type == "hybrid":
            loss_recon = self._reconstruction_loss(x_hat, x0)
            loss = self.drift_weight * loss_drift + self.recon_weight * loss_recon
            if mr_loss is not None:
                loss = loss + self.mr_stft_weight * mr_loss
            self.log("drift_loss", loss_drift, on_step=True, on_epoch=True)
            self.log("recon_loss", loss_recon, on_step=True, on_epoch=True)
            # Monitoring: ratio and V magnitude
            with torch.no_grad():
                ratio = loss_drift / (loss_recon + 1e-8)
            self.log("drift_recon_ratio", ratio, on_step=True, on_epoch=True)
        else:
            # Pure drifting loss
            loss = self.drift_weight * loss_drift
            if mr_loss is not None:
                loss = loss + self.mr_stft_weight * mr_loss
            self.log("drift_loss", loss_drift, on_step=True, on_epoch=True)

        self.log("drift_v_magnitude", v_mag, on_step=True, on_epoch=True)

        return loss

    # ========== Training / Validation Steps ==========

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement metrics
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_val, si_sdr_val, estoi_val = self._evaluate()
            self.log("pesq", pesq_val, on_step=False, on_epoch=True)
            self.log("si_sdr", si_sdr_val, on_step=False, on_epoch=True)
            self.log("estoi", estoi_val, on_step=False, on_epoch=True)

        return loss

    def _evaluate(self):
        """Evaluate model on validation files using single-step inference."""
        _import_metrics()
        import soundfile as sf
        import torchaudio

        sr = 16000
        clean_files = self.data_module.valid_set.clean_files
        noisy_files = self.data_module.valid_set.noisy_files

        total_num_files = len(clean_files)
        num_eval = min(self.num_eval_files, total_num_files)
        indices = torch.linspace(0, total_num_files - 1, num_eval, dtype=torch.int)
        clean_files = [clean_files[i] for i in indices]
        noisy_files = [noisy_files[i] for i in indices]

        total_pesq = 0
        total_si_sdr = 0
        total_estoi = 0

        for clean_file, noisy_file in zip(clean_files, noisy_files):
            # Load and resample audio
            x_wav, x_sr = sf.read(clean_file)
            x_wav = torch.from_numpy(x_wav).float()
            if x_wav.dim() == 1:
                x_wav = x_wav.unsqueeze(0)
            if x_sr != sr:
                x_wav = torchaudio.functional.resample(x_wav, x_sr, sr)

            y_wav, y_sr = sf.read(noisy_file)
            y_wav = torch.from_numpy(y_wav).float()
            if y_wav.dim() == 1:
                y_wav = y_wav.unsqueeze(0)
            if y_sr != sr:
                y_wav = torchaudio.functional.resample(y_wav, y_sr, sr)

            T_orig = x_wav.size(1)

            # Normalize
            norm_factor = y_wav.abs().max()
            y_wav_norm = y_wav / norm_factor

            # STFT + transform
            Y = torch.unsqueeze(
                self._forward_transform(self._stft(y_wav_norm.to(self.device))), 0
            )
            Y = pad_spec(Y)

            # Single-step enhancement (no ODE integration!)
            with torch.no_grad():
                x_hat_spec = self.forward(Y)

            # Convert back to audio
            x_hat_wav = self.to_audio(x_hat_spec.squeeze(), T_orig)
            x_hat_wav = x_hat_wav * norm_factor

            x_hat_np = x_hat_wav.squeeze().cpu().numpy()
            x_np = x_wav.squeeze().cpu().numpy()

            total_si_sdr += si_sdr(x_np, x_hat_np)
            total_pesq += _pesq(sr, x_np, x_hat_np, "wb")
            total_estoi += _stoi(x_np, x_hat_np, sr, extended=True)

        return (
            total_pesq / num_eval,
            total_si_sdr / num_eval,
            total_estoi / num_eval,
        )

    # ========== Data Module Interface ==========

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
