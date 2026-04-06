from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
from safetensors.torch import save_file

from model import MiniLLM
from tokenizer import load_tokenizer
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")# TODO: allow user to specify device
    model = MiniLLM(
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        rotary_base=cfg["model"]["rotary_base"],
        rotary_scaling_factor=cfg["model"]["rotary_scaling_factor"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        pad_token_id=cfg["model"]["pad_token_id"],
    ).to(device)

    load_checkpoint(args.checkpoint, model, map_location="cpu")

    save_file(model.state_dict(), str(output / "model.safetensors"))

    hf_config = {
        "architectures": ["MiniLLM"],
        "model_type": "mini_llm",
        "vocab_size": cfg["model"]["vocab_size"],
        "max_position_embeddings": cfg["model"]["max_seq_len"],
        "hidden_size": cfg["model"]["d_model"],
        "num_hidden_layers": cfg["model"]["n_layers"],
        "num_attention_heads": cfg["model"]["n_heads"],
        "num_key_value_heads": cfg["model"]["n_kv_heads"],
        "intermediate_size": cfg["model"]["d_ff"],
        "tie_word_embeddings": cfg["model"]["tie_embeddings"],
        "pad_token_id": cfg["model"]["pad_token_id"],
        "bos_token_id": cfg["model"]["bos_token_id"],
        "eos_token_id": cfg["model"]["eos_token_id"],
    }
    (output / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")

    tok = load_tokenizer(cfg["data"]["tokenizer_path"])
    tok.save(str(output / "tokenizer.json"))

    print(f"Saved HF-style export to {output}")


if __name__ == "__main__":
    main()
