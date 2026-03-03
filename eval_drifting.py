"""
Evaluation script for DriftingSE model.
Single-step inference (1-NFE) — no ODE integration needed.

Usage:
  python eval_drifting.py --checkpoint <path> --data_dir <dataset_dir> --split test
"""

import argparse
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pesq import pesq
from pystoi import stoi
from glob import glob

from flowmse.drifting_model import DriftingSEModel
from flowmse.util.other import si_sdr, pad_spec


def load_audio(filepath, target_sr=16000):
    waveform, sample_rate = sf.read(filepath)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def evaluate(args):
    sr = 16000

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = DriftingSEModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    model.cuda()

    # Get file lists
    clean_dir = os.path.join(args.data_dir, args.split, "clean")
    noisy_dir = os.path.join(args.data_dir, args.split, "noisy")
    clean_files = sorted(glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))
    assert len(clean_files) == len(noisy_files), "Mismatched clean/noisy files"
    print(f"Evaluating on {len(clean_files)} files from {args.split} split")

    # Output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    metrics = {"pesq": [], "si_sdr": [], "estoi": []}

    for i, (cf, nf) in enumerate(zip(clean_files, noisy_files)):
        # Load audio
        x = load_audio(cf, sr)
        y = load_audio(nf, sr)
        T_orig = x.size(1)

        # Normalize
        norm_factor = y.abs().max()
        y_norm = y / norm_factor

        # STFT + transform
        Y = torch.unsqueeze(
            model._forward_transform(model._stft(y_norm.cuda())), 0
        )
        Y = pad_spec(Y)

        # Single-step enhancement (1-NFE)
        with torch.no_grad():
            x_hat_spec = model.forward(Y)

        # Convert to waveform
        x_hat = model.to_audio(x_hat_spec.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat_np = x_hat.squeeze().cpu().numpy()
        x_np = x.squeeze().numpy()

        # Compute metrics
        p = pesq(sr, x_np, x_hat_np, "wb")
        s = si_sdr(x_np, x_hat_np)
        e = stoi(x_np, x_hat_np, sr, extended=True)

        metrics["pesq"].append(p)
        metrics["si_sdr"].append(s)
        metrics["estoi"].append(e)

        # Save enhanced audio
        if args.output_dir:
            fname = os.path.basename(nf)
            sf.write(os.path.join(args.output_dir, fname), x_hat_np, sr)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(clean_files)}] "
                  f"PESQ={p:.3f} SI-SDR={s:.2f} ESTOI={e:.3f}")

    # Print summary
    print("\n=== Results (DriftingSE, 1-NFE) ===")
    for k, v in metrics.items():
        print(f"  {k.upper():>8s}: {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save enhanced audio files")
    args = parser.parse_args()
    evaluate(args)
