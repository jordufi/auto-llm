from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import MiniLLM
from tokenizer import load_tokenizer
from utils.checkpoint import load_checkpoint
from utils.chat_format import format_chat
from utils.sampling import sample_next_token
from utils.config import load_config


@torch.no_grad()
def chat_loop(
    model,
    tokenizer,
    device,
    amp_dtype,
    system_prompt,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    max_seq_len,
    mode="chat",
):
    history = []
    if mode == "chat":
        print(
            "Enter your message. Type /exit to quit, /reset to clear the conversation."
        )
    else:
        print(
            "Enter a text prompt for completion. Type /exit to quit, /reset to clear."
        )
    while True:
        user = input("\nYou: ").strip()
        if user == "/exit":
            break
        if user == "/reset":
            history = []
            continue

        if mode == "chat":
            history.append({"role": "user", "content": user})
            prompt = format_chat(history, system_prompt=system_prompt)
        else:
            prompt = user

        ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        model.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -max_seq_len:]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype in (torch.float16, torch.bfloat16),
            ):
                out = model(input_ids=ctx)
            logits = out.logits[:, -1, :]
            next_id = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == 2:
                break

        if mode == "chat":
            text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
            answer = text.split("<|assistant|>")[-1].strip()
            print(f"Assistant: {answer}")
            history.append({"role": "assistant", "content": answer})
        else:
            text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            print(f"Completion: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=["chat", "base"],
        default=None,
        help="chat=use chat format, base=plain text completion. Auto-detected from config if not set.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16 if cfg["training"]["precision"] == "bf16" else torch.float16
    )

    # Auto-detect mode from config if not explicitly set
    if args.mode is not None:
        mode = args.mode
    elif cfg["training"].get("generation_use_chat", True):
        mode = "chat"
    else:
        mode = "base"

    tokenizer = load_tokenizer(cfg["data"]["tokenizer_path"])
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

    load_checkpoint(args.checkpoint, model, map_location=device)

    chat_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        system_prompt=cfg["data"]["chat_template"]["system"],
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=cfg["model"]["max_seq_len"],
        mode=mode,
    )


if __name__ == "__main__":
    main()
