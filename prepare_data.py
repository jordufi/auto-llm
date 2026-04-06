from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.config import load_config
from dataset import build_token_cache


def split_token_cache(
    train_cache_path: str, val_cache_path: str, val_ratio: float = 0.00002
):
    """Split a flat token cache into train and val by holding out a percentage of tokens."""
    train_path = Path(train_cache_path)
    val_path = Path(val_cache_path)

    if val_path.exists():
        return

    data = np.memmap(train_path, dtype=np.uint32, mode="r")
    total = len(data)
    val_size = max(1024, int(total * val_ratio))
    split_point = total - val_size

    # Copy val portion to separate file
    val_tokens = np.array(data[split_point:], dtype=np.uint32)
    val_tokens.tofile(val_path)

    # Rewrite train without val portion
    train_tokens = np.array(data[:split_point], dtype=np.uint32)
    train_tokens.tofile(train_path)

    print(
        f"Split: {split_point:,} train tokens, {val_size:,} val tokens ({val_ratio*100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    build_token_cache(
        raw_paths=data_cfg["raw_train_paths"],
        tokenizer_path=data_cfg["tokenizer_path"],
        out_path=data_cfg["train_cache"],
        dtype=data_cfg.get("cache_dtype", "uint32"),
        overwrite=data_cfg.get("overwrite_cache", False),
        system_prompt=data_cfg["chat_template"]["system"],
    )

    val_ratio = data_cfg.get("val_split_ratio", 0.0)

    if data_cfg.get("raw_val_paths"):
        build_token_cache(
            raw_paths=data_cfg["raw_val_paths"],
            tokenizer_path=data_cfg["tokenizer_path"],
            out_path=data_cfg["val_cache"],
            dtype=data_cfg.get("cache_dtype", "uint32"),
            overwrite=data_cfg.get("overwrite_cache", False),
            system_prompt=data_cfg["chat_template"]["system"],
        )
    elif val_ratio > 0:
        split_token_cache(
            train_cache_path=data_cfg["train_cache"],
            val_cache_path=data_cfg["val_cache"],
            val_ratio=val_ratio,
        )


if __name__ == "__main__":
    main()
