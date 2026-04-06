import argparse
import json
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get("text", "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < 200:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"Wrote {written} rows to {args.out}")

if __name__ == "__main__":
    main()