# Mini LLM 2026

Single-GPU mini LLM training pipeline for a YouTube build.

TALK ABOUT CHINCHILLA POINT

## What is included
- Decoder-only Transformer with:
  - RMSNorm
  - SwiGLU
  - RoPE
  - FlashAttention-equivalent via PyTorch SDPA
  - MTP-lite loss for t+1 and t+2 prediction
- Tokenizer training and loading
- Data cache builder
- Training with checkpoint save/resume
- TensorBoard logging with text samples
- CLI chatbot inference
- Export helper for safetensors / HF-style folder



## conda environment
```bash
conda create -n mini-llm python=3.10 -y
conda activate mini-llm
```

## pytorch version
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## pytorch version for rtx 50 series
```bash
pip uninstall torch torchvision torchaudio -y

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## check it is installed correctly
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Install
```bash
pip install -r requirements.txt
```

## flash attention -> not for windows
```bash
pip install flash-attn --no-build-isolation
```
# for windows -> 
torch.nn.functional.scaled_dot_product_attention
instead of flash attention

## Download the base datasets
```bash 
#python scripts/build_base_corpus.py --dataset Skylion007/openwebtext --out data/base_openwebtext.jsonl --limit 200000


python scripts/build_base_corpus.py --dataset Skylion007/openwebtext --out data/base_openwebtext.jsonl --limit 5000000
#python scripts/build_base_corpus.py --dataset roneneldan/TinyStories --out data/base_tinystories.jsonl --limit 1000000
```

## Download the chat datasets
```bash 
python scripts/build_chat_corpus.py --dataset openbmb/UltraChat --out data/chat_ultrachat.jsonl --limit 200000
python scripts/build_chat_corpus.py --dataset teknium/openhermes --out data/chat_openhermes.jsonl --limit 200000
```bash 



## 1) Train tokenizer
```bash

python tokenizer.py train --input data/base_openwebtext.jsonl --output artifacts/tokenizer.json --vocab-size 50000

```


## 2) Build token caches
```bash
python prepare_data.py --config config.yaml
```



## 3) Train
```bash
python train.py --config config.yaml
```

## 4) Resume
```bash
python train.py --config config.yaml --resume outputs/mini_llm_2026/checkpoints/step_1000.pt
```

## 5) Chat
```bash
python inference.py --config config.yaml --checkpoint outputs/mini_llm_2026/checkpoints/last.pt
```

## TensorBoard
```bash
tensorboard --logdir outputs/mini_llm_2026/tensorboard
```

## Inference: 
# 4) For inference on the base model (text completion, not chat)
python inference.py --config config-base-model-110m.yaml --checkpoint path/to/checkpoint.pt
# It will auto-detect base mode. Or explicitly:
python inference.py --config config-base-model-110m.yaml --checkpoint path/to/checkpoint.pt --mode base
```

## Notes
- The code uses PyTorch scaled_dot_product_attention, which dispatches to flash kernels when available.
- For one GPU, start with micro_batch_size=1 and grad_accum_steps=8.
- If VRAM is tight, reduce `d_model`, `n_layers`, or `max_seq_len`.

## Export
This repo includes a lightweight export script for safetensors/HF-style folders. For llama.cpp / LM Studio, you will usually need to convert through a HuggingFace-compatible checkpoint or use a custom converter for this architecture.
