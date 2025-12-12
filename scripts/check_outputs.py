#!/usr/bin/env python3
import torch
import os
import sys
import glob

from config import load_args
from dynamics_learning.lighting import DynamicsLearning


def infer_output_size_from_checkpoint(state_dict):
    """
    Infer output dimension from the final Linear layer in the decoder.
    This works for your repo because decoder ends with nn.Linear(..., output_size).
    """
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith("weight") and v.ndim == 2:
            linear_weights.append((k, v.shape))

    # The LAST linear layer corresponds to the output head
    last_weight_name, last_weight_shape = linear_weights[-1]
    output_size = last_weight_shape[0]
    return output_size, last_weight_name


def inspect_model(exp_path, ckpt_path):
    print("=" * 70)
    print(f"Experiment path : {exp_path}")
    print(f"Checkpoint path : {ckpt_path}")

    args = load_args(os.path.join(exp_path, "args.txt"))
    print("predictor_type  :", args.predictor_type)
    print("history_length :", args.history_length)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    output_size, layer_name = infer_output_size_from_checkpoint(state_dict)
    print("inferred output_size :", output_size)
    print("output layer         :", layer_name)

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"

    model = DynamicsLearning(
        args,
        resources_path,
        exp_path,
        input_size=14,
        output_size=output_size,
        max_iterations=1,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    H = args.history_length
    x = torch.zeros(H * 14)

    with torch.no_grad():
        y = model(x)

    print("raw output tensor shape :", tuple(y.shape))
    print("raw output numel       :", y.numel())
    print("=" * 70)
    print()


def find_latest_checkpoint(exp_path):
    ckpts = glob.glob(os.path.join(exp_path, "checkpoints", "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {exp_path}/checkpoints/")
    return max(ckpts, key=os.path.getctime)


if __name__ == "__main__":
    base = "/workspace/resources/experiments/"

    velocity_exp = os.path.join(base, "velocity")
    attitude_exp = os.path.join(base, "attitude")

    velocity_ckpt = find_latest_checkpoint(velocity_exp)
    attitude_ckpt = find_latest_checkpoint(attitude_exp)

    inspect_model(velocity_exp, velocity_ckpt)
    inspect_model(attitude_exp, attitude_ckpt)
