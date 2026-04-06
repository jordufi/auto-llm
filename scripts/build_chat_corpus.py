import argparse
import json
from datasets import load_dataset

SYSTEM_PROMPT = "You are a helpful assistant."


def to_chat_text(example):
    SYSTEM_PROMPT = "You are a helpful assistant."

    # 🟢 UltraChat format
    if "conversations" in example:
        dialogue = example["conversations"]

        if len(dialogue) < 2:
            return None

        user_msg = dialogue[0]
        assistant_msg = dialogue[1]

        if user_msg.get("from") != "human" or assistant_msg.get("from") != "gpt":
            return None

        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user_msg['value']}\n"
            f"<|assistant|>\n{assistant_msg['value']}"
        )

    # 🔵 OpenHermes format
    if "instruction" in example and "output" in example:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if not isinstance(output, str):
            return None

        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n{output}"
        )

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            text = to_chat_text(ex)
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"Wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
