"""Pack ZLM parameters in native decode order as raw BF16."""
import argparse
from pathlib import Path

import torch

from scripts.model import ModelConfig


def export(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    weights = state.get("model", state)
    suffixes = ("attention_norm", "attention.q_proj", "attention.k_proj",
                "attention.v_proj", "attention.o_proj", "mlp_norm", "mlp.gate_proj",
                "mlp.up_proj", "mlp.down_proj")
    names = ["embed_tokens.weight", "norm.weight"]
    for layer in range(ModelConfig().num_layers):
        names.extend(f"layers.{layer}.{suffix}.weight" for suffix in suffixes)
    with open(output, "wb") as stream:
        for name in names:
            weights[name].to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tofile(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
