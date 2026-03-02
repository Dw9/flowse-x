#!/usr/bin/env python
"""Full test-set evaluation for flowmse checkpoints with 16kHz resampling."""
import argparse
import torch
import torch.serialization
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from flowmse.data_module import SpecsDataModule, load_audio
from flowmse.model import VFModel
from flowmse.sampling import get_white_box_solver
from flowmse.util.other import si_sdr, pad_spec

torch.serialization.add_safe_globals([SpecsDataModule])

SR = 16000
INFERENCE_N = 5


def evaluate_checkpoint(ckpt_path, data_dir, split="test", num_files=None):
    print(f"Loading checkpoint: {ckpt_path}")
    model = VFModel.load_from_checkpoint(
        ckpt_path, base_dir=data_dir, map_location="cpu"
    )
    model.data_module.setup(stage=None)

    # Swap to EMA weights
    for name, param in model.dnn.named_parameters():
        if name in model.ema_dnn:
            param.data = model.ema_dnn[name].to(param.device)

    model.eval()
    model.cuda()

    T_rev = model.T_rev
    t_eps = model.t_eps

    if split == "test":
        clean_files = model.data_module.test_set.clean_files
        noisy_files = model.data_module.test_set.noisy_files
    else:
        clean_files = model.data_module.valid_set.clean_files
        noisy_files = model.data_module.valid_set.noisy_files

    if num_files:
        indices = torch.linspace(0, len(clean_files)-1, num_files, dtype=torch.int)
        clean_files = [clean_files[i] for i in indices]
        noisy_files = [noisy_files[i] for i in indices]

    pesq_scores, si_sdr_scores, estoi_scores = [], [], []
    errors = 0

    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        try:
            x, _ = load_audio(clean_file)  # Now resamples to 16kHz
            y, _ = load_audio(noisy_file)
            T_orig = x.size(1)

            norm_factor = y.abs().max()
            y_norm = y / norm_factor

            Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.cuda())), 0)
            Y = pad_spec(Y)

            with torch.no_grad():
                sampler = get_white_box_solver(
                    "euler", model.ode, model, Y.cuda(),
                    T_rev=T_rev, t_eps=t_eps, N=INFERENCE_N
                )
                sample, _ = sampler()

            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor

            x_np = x.squeeze().cpu().numpy()
            x_hat_np = x_hat.squeeze().cpu().numpy()

            pesq_scores.append(pesq(SR, x_np, x_hat_np, "wb"))
            si_sdr_scores.append(si_sdr(x_np, x_hat_np))
            estoi_scores.append(stoi(x_np, x_hat_np, SR, extended=True))
        except Exception as e:
            errors += 1
            print(f"Error on {clean_file}: {e}")

    n = len(pesq_scores)
    print(f"\n{'='*50}")
    print(f"Results on {split} set ({n} files, {errors} errors)")
    print(f"{'='*50}")
    print(f"PESQ:  {np.mean(pesq_scores):.4f} ± {np.std(pesq_scores):.4f}")
    print(f"SI-SDR: {np.mean(si_sdr_scores):.4f} ± {np.std(si_sdr_scores):.4f}")
    print(f"ESTOI: {np.mean(estoi_scores):.4f} ± {np.std(estoi_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/home/zhibo/workspace/VoiceBank_processed")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--num_files", type=int, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.data_dir, args.split, args.num_files)
