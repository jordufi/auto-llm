from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from tokenizer import load_tokenizer
from utils.chat_format import format_chat

# Enable parallel tokenization in the Rust backend
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def extract_text_from_record(record: dict, system_prompt: str = "You are a helpful assistant.") -> str:
    if "text" in record and record["text"]:
        return str(record["text"])
    if "messages" in record and record["messages"]:
        return format_chat(record["messages"], system_prompt=system_prompt)
    return ""

def iter_raw_texts(paths: Sequence[str], system_prompt: str = "You are a helpful assistant.") -> Iterable[str]:
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: Path {p} not found. Skipping.")
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        text = extract_text_from_record(rec, system_prompt=system_prompt)
                        if text: yield text
                    except json.JSONDecodeError:
                        continue
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line: yield line

def build_token_cache(
    raw_paths: Sequence[str],
    tokenizer_path: str,
    out_path: str,
    system_prompt: str = "You are a helpful assistant.",
    dtype: str = "uint32",
    overwrite: bool = False,
    batch_size: int = 15000,
    max_tokens: int = 3_000_000_000, # Artificial Limit
) -> str:
    out = Path(out_path)
    if out.exists() and not overwrite:
        print(f"Cache {out_path} already exists. Skipping.")
        return str(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(tokenizer_path)

    total_processed_tokens = 0
    print(f"Starting Streaming Tokenization...")
    print(f"Target: {max_tokens:,} tokens | Batch Size: {batch_size}")

    # Ensure we start with a fresh file
    if out.exists():
        out.unlink()

    current_batch = []
    for text in iter_raw_texts(raw_paths, system_prompt=system_prompt):
        current_batch.append(text)
        
        if len(current_batch) >= batch_size:
            # Tokenize the batch
            encodings = tok.encode_batch(current_batch, add_special_tokens=False)
            
            # Flatten and convert to numpy array immediately
            batch_ids = []
            for enc in encodings:
                batch_ids.extend(enc.ids)
            
            arr = np.array(batch_ids, dtype=getattr(np, dtype))
            
            # Append directly to the file on disk
            with open(out, "ab") as f:
                arr.tofile(f)
            
            total_processed_tokens += len(batch_ids)
            current_batch = []
            
            print(f"Progress: {total_processed_tokens:,} / {max_tokens:,} tokens", end='\r')

            # Check the limit
            if total_processed_tokens >= max_tokens:
                print(f"\nLimit reached: {total_processed_tokens:,} tokens saved.")
                return str(out)

    # Final partial batch
    if current_batch:
        encodings = tok.encode_batch(current_batch, add_special_tokens=False)
        batch_ids = [tid for enc in encodings for tid in enc.ids]
        if batch_ids:
            arr = np.array(batch_ids, dtype=getattr(np, dtype))
            with open(out, "ab") as f:
                arr.tofile(f)
            total_processed_tokens += len(batch_ids)

    print(f"\nFinished. Total tokens: {total_processed_tokens:,}")
    return str(out)

class RandomTokenBatchStream(IterableDataset):
    def __init__(self, cache_path: str, seq_len: int, batch_size: int, pad_token_id: int = 0):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        # memmap is safe to use here because we are only reading
        self.data = np.memmap(self.cache_path, dtype=np.uint32, mode="r")

    def __iter__(self):
        rng = np.random.default_rng()
        total_len = len(self.data)
        need = self.seq_len + 1
        while True:
            batch = []
            for _ in range(self.batch_size):
                if total_len <= need + 1:
                    start = 0
                else:
                    start = int(rng.integers(0, total_len - need - 1))
                chunk = np.array(self.data[start : start + need], dtype=np.int64)
                if len(chunk) < need:
                    pad = np.full(need - len(chunk), self.pad_token_id, dtype=np.int64)
                    chunk = np.concatenate([chunk, pad], axis=0)
                batch.append(torch.tensor(chunk, dtype=torch.long))
            yield torch.stack(batch, dim=0)