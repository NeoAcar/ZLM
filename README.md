# ZLM

Code for Trinity, a Turkish-first 0.5B ZLM model pretrained from scratch and instruction fine-tuned for chat,
with a native BF16 CUDA decoder.

| | |
| --- | --- |
| Model (instruct, pretrained base, tokenizer) | [137N/trinity](https://huggingface.co/137N/trinity) |
| Pretraining data | [137N/trinity-turkish-pretrain](https://huggingface.co/datasets/137N/trinity-turkish-pretrain) |
| SFT data | [137N/trinity-turkish-sft](https://huggingface.co/datasets/137N/trinity-turkish-sft) |

## Setup

Linux x86_64 with an NVIDIA GPU, Python 3.13.16 and PyTorch 2.14.1+cu130, managed with uv:

```bash
uv python install 3.13.16
uv sync --locked
```

Training logs to Weights & Biases (project `ZLM`); log in with `uv run wandb login` first.

## Model

`scripts/model.py`: 508,638,208 parameters, 38 layers, width 1024, 16 query / 4 KV heads (GQA),
SwiGLU 3072, RMSNorm, tied embeddings, 49,152-token vocabulary, RoPE base 100,000 with a 4,096-token
context and a 16,384-token YaRN phase. Linear matrices use Muon; the embedding and norms use AdamW.

## Tokenizer

The released tokenizer is `tokenizer/tokenizer-49152.json` (Hugging Face `tokenizers` format). Load it
with `Tokenizer.from_file(...)` and set `encode_special_tokens = True` (not stored in the file) so text
never turns into control tokens.

It is a byte-level BPE trained on Parquet shards with a `text` column:

```bash
uv run python scripts/train_tokenizer.py --train SAMPLE_DIR --output tokenizer-49152.json
```

IDs 0-127 are control tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`, roles, `<|turn_end|>`, ...,
`<|reserved_114|>`); SFT uses `<|reserved_0|>` (13) as turn start.

## Pretraining

Download the token data from [137N/trinity-turkish-pretrain](https://huggingface.co/datasets/137N/trinity-turkish-pretrain)
(about 24 GB) and train:

```bash
uv run hf download 137N/trinity-turkish-pretrain --repo-type dataset \
  --include "tokens/*" "academic-decay/*" --local-dir data
uv run python scripts/train_model.py --data data/tokens --decay-data data/academic-decay \
  --output outputs/pretrain [--resume outputs/pretrain/checkpoint-XXXXXXXX.pt]
```

Token directories hold little-endian uint16 shards and a `manifest.json`. Validation uses the last 4,598
blocks of the shuffled main corpus, which training never reaches. The schedule is the one that produced the
released base checkpoint (step 22,606, 11B tokens):

| Steps | Context | Global batch | LR (Muon / AdamW peak) |
| --- | --- | --- | --- |
| 0-499 | 4,096 | 262,144 | linear warmup to 1e-3 / 6e-4 |
| 500-3,249 | 4,096 | 262,144 | constant |
| 3,250-9,418 | 4,096 | 524,288 | constant |
| 9,419-14,365 | 4,096 | 524,288 | x 2/3 |
| 14,366-20,697 | 4,096 | 524,288 | x 1/3 |
| 20,698-22,605 | 16,384 (YaRN) | 524,288 | academic decay, linear to zero |

Checkpoints are written every 30 minutes and at both phase ends; the latest two are kept.

## Chat fine-tuning (SFT)

Download [137N/trinity-turkish-sft](https://huggingface.co/datasets/137N/trinity-turkish-sft) and the base model
from [137N/trinity](https://huggingface.co/137N/trinity), then train:

```bash
uv run hf download 137N/trinity-turkish-sft --repo-type dataset --local-dir sft
uv run hf download 137N/trinity pretrained/checkpoint-00022606.pt --local-dir trinity
uv run python scripts/train_sft.py --base trinity/pretrained/checkpoint-00022606.pt \
  --data sft/data/train.parquet --tokenizer-path tokenizer/tokenizer-49152.json --output outputs/sft
```

A seed-137 random 1,499 dialogues are held out for validation. Template: `<|bos|>` then
`<|turn_start|> role text <|turn_end|>` for every message; the loss covers only assistant text and its
`<|turn_end|>`. AdamW, peak LR 5e-5, 138 warmup steps, cosine to 1e-6, 64 dialogues per step, 2 epochs.
`<|turn_end|>` starts from the pretrained `<|eos|>` embedding plus 10% noise.

## Native BF16 decoding

`inference/cuda/decode.cuh` is a single-request CUDA decoder with a KV cache: about 202 tok/s on an
RTX 3050 Ti Laptop at a 512-token depth, versus 78 tok/s for llama.cpp on the same card. It runs on
NVIDIA GPUs of compute capability 8.0 or newer (Ampere and later).

1. Pack a checkpoint into the decoder's raw BF16 layout:

   ```bash
   uv run python -m inference.cuda.export CHECKPOINT.pt model.bin   # or download instruct/model.bin from 137N/trinity
   ```

2. Build the shared library for Python and, optionally, the throughput benchmark:

   ```bash
   nvcc -O3 -std=c++17 -arch=compute_80 -code=sm_86,compute_80 -shared -Xcompiler -fPIC \
     inference/cuda/capi.cu -o libzlm_decode.so
   nvcc -O3 -std=c++17 -arch=compute_80 -code=sm_86,compute_80 inference/cuda/bench.cu -o bench
   ./bench model.bin
   ```

3. Decode from Python (the SFT model uses the 16,384-token YaRN phase):

   ```python
   from inference.cuda.native import NativeDecoder

   decoder = NativeDecoder('libzlm_decode.so', 'model.bin', capacity=16384, yarn=True)
   for token in prompt_ids[:-1]:
       decoder.feed(token)              # fill the KV cache
   logits = decoder.step(prompt_ids[-1])  # FP32 numpy logits for the next token
   answer = []
   while (token := int(logits.argmax())) != 8:  # greedy until <|turn_end|> (8)
       answer.append(token)
       logits = decoder.step(token)
   decoder.length = 0                   # rewind the cache for a new conversation
   ```

   The chat prompt format is described under *Chat fine-tuning*. From C++, include `decode.cuh` and
   use `zlm::Decoder` directly; `capi.cu` shows the calls.
