"""Multi-resolution STFT loss for waveform-domain supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    Computes STFT at multiple resolutions on waveforms, penalizing spectral
    differences at each scale. Standard in speech synthesis/enhancement
    (HiFi-GAN, Vocos, EnCodec).

    Args:
        resolutions: List of (n_fft, hop_length, win_length) tuples.
        factor_sc: Weight for spectral convergence loss.
        factor_mag: Weight for log-magnitude L1 loss.
    """

    DEFAULT_RESOLUTIONS = [
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    ]

    def __init__(self, resolutions=None, factor_sc=1.0, factor_mag=1.0):
        super().__init__()
        self.resolutions = resolutions or self.DEFAULT_RESOLUTIONS
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        for n_fft, _, win_length in self.resolutions:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(win_length))

    def _get_window(self, n_fft, device):
        window = getattr(self, f"window_{n_fft}")
        if window.device != device:
            window = window.to(device)
        return window

    def forward(self, x_hat_wav, x_wav):
        """Compute multi-resolution STFT loss.

        Args:
            x_hat_wav: [B, T] predicted waveform.
            x_wav: [B, T] target waveform.

        Returns:
            Scalar loss averaged over resolutions.
        """
        total_loss = 0.0
        for n_fft, hop_length, win_length in self.resolutions:
            window = self._get_window(n_fft, x_hat_wav.device)

            pred_stft = torch.stft(
                x_hat_wav, n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )
            target_stft = torch.stft(
                x_wav, n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Spectral convergence: Frobenius norm ratio
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-7
            )

            # Log-magnitude L1
            mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7)
            )

            total_loss = total_loss + self.factor_sc * sc_loss + self.factor_mag * mag_loss

        return total_loss / len(self.resolutions)
