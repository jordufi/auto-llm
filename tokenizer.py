from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]


def iter_texts(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".jsonl":
            import json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if "text" in obj and obj["text"]:
                        yield str(obj["text"])
                    elif "messages" in obj and obj["messages"]:
                        from utils.chat_format import format_chat
                        yield format_chat(obj["messages"])
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def train_tokenizer(input_paths: list[str], output_path: str, vocab_size: int = 50000) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_texts(input_paths), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_text(tokenizer: Tokenizer, text: str, add_special_tokens: bool = True):
    if add_special_tokens:
        return tokenizer.encode(text).ids
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode_ids(tokenizer: Tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--vocab-size", type=int, default=50000)

    args = parser.parse_args()
    if args.cmd == "train":
        train_tokenizer(args.input, args.output, args.vocab_size)


if __name__ == "__main__":
    main()
