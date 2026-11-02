# KnowLine-31B-Gen1: serving and Decision Index reproduction

These settings reproduce our self-run Decision Index 0.3 evaluation of KnowLine-31B-Gen1: the full 0.3 public suite sent
fresh (no rows carried over from another run) through the release front end with read twice on, scored 2026-10-24 04:28 CST.

## Weights

- bf16, about 63 GB. They were made in three steps:
  1. **Fine-tuning:** gemma-4-31B-it trained for one epoch with a LoRA adapter (rank 32, alpha 64, language model
     only), merged into the weights.
  2. **Averaging with the base model:** every tensor is 0.75 × the fine-tuned tensor + 0.25 × the gemma-4-31B-it tensor
     (computed in fp32, stored in bf16). Tensors that fine-tuning did not change, such as the vision tower, stay those
     of the base model.
  3. **Readout temperature T = 1.44**, folded in two places so that every final logit is divided exactly by T:
     - the final norm weight `model.language_model.norm.weight` is divided by 1.44 (Gemma 4's RMSNorm applies the
       weight as `x * w`, not `(1 + w) * x`);
     - `text_config.final_logit_softcapping` in `config.json` is 20.833333333333336 instead of 30.0. Gemma 4 caps its logits as
       `cap * tanh(z / cap)`; with both changes the output is `30 * tanh(z / 30) / T`, the capped logit divided by T.
       **Do not change the soft-cap back to 30**, or the temperature is only partly applied.
     - T was fitted by negative log-likelihood on our in-training evaluation sets, served read twice through this front
       end (4,965 questions); refitting on the released weights gives T = 1.00, and ECE on our held-out sets (6,995
       questions) is 0.044 (0.102 at T = 1 before folding).
- The architecture is `Gemma4ForConditionalGeneration`. Apart from `final_logit_softcapping`, `config.json` is identical
  to that of [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it); the tokenizer files, the chat
  template, `processor_config.json` and `generation_config.json` are byte-identical to the base model.
- `SHA256SUMS` lists every file.

## Software

| component | version |
|---|---|
| SGLang | 0.5.21 (torch 2.13.0, CUDA 13.0) |
| transformers | 5.12.1 |
| front end | `knowline_server.py` + `knowline_layout.py` (this repository; need only transformers and requests) |
| decision-index kit | 0.3 at commit 62d2f51de34a2de64906345b6bc3e98e27ff55c7 |

### About the front end

`knowline_server.py` is the `/v1/systemone` front end of our runs; the prompt layout lives in `knowline_layout.py` next to
it. Text requests are rendered exactly as by the KnowLine-26B-A4B-Gen2.5 front end, prompt for prompt (checked on every
backend call of our test requests). It renders with the model's
own chat template. The front end contains:

- rendering: the chat template with thinking off; the state is followed by one user turn with the instruction, the
  question and all options labelled A, B, ...; the assistant turn opens with `Answer:`;
- read twice (on by default, see below);
- label-token scoring: one prefill per question, then a softmax over the option labels only;
- the HTTP server.

### Read twice

- **What it does:** for a text-only request of at most 3,000 tokens (counted with the model's tokenizer over the whole
  request read once), every turn after the system turn is written a second time before the answer is read: the state, the
  instruction, the question and its options. The second copy starts with the line "Read the same input again before
  answering:". The answer is still read from the label token after `Answer:`, one prefill per question.
- **What it leaves alone:** requests over 3,000 tokens and requests with images are sent once. In our Decision Index run,
  132,376 of the 140,620 requests (94.1%) were read twice (the rows and the Gemma 4 tokenizer are those of
  KnowLine-26B-A4B-Gen2.5's run, so the same requests fall under the cut).
- **Switches:** `--no-read-twice` reads every request once; `--twice-max-tokens` sets the cut. `knowline_engine.py` takes
  `--option read_twice=0` and `--option twice_max_tokens=<n>` for the same. All our evaluations of this model used read
  twice.
- **Cost:** about 1.54 times the prompt tokens of reading once over the whole suite (323 million against 209 million, the
  same rows and tokenizer as KnowLine-26B-A4B-Gen2.5's run); see
  [Hardware and latency](#hardware-and-latency).
- **Credit:** the read-twice layout follows FlexDM ([github.com/1000luca/FlexDM](https://github.com/1000luca/FlexDM), MIT),
  including its "Read the same input again before answering:" line; repeating the input goes back to RE2 (Xu et al.,
  2023, [arXiv:2309.06275](https://arxiv.org/abs/2309.06275)). `knowline_layout.py` is our own implementation.

The front end uses temperature 1 and the readout temperature is in the weights, so both backends get it. One setting
is new in this release: `temperatures.json` (`{"score": 0.6944}`, shipped in this repository) makes the front end divide
the logits of score (rubric) questions by a further 0.6944, so they are read at an effective temperature of
1.44 × 0.6944 = 1.0, while yes/no and choice questions stay at 1.44. Score questions are often read as an expected value
over the scale, which a higher temperature pulls toward the middle. `serve_knowline.sh` passes the file and
`knowline_engine.py` loads it by default (`--option temperatures=none` turns it off). The earlier KnowLine releases did
not use per-type temperatures. All our evaluations of this model used them, except where noted.

## Serve

`serve_knowline.sh` starts both processes, with the settings of KnowLine-26B-A4B-Gen2.5 plus `temperatures.json` and
the box-question flags.

1. **SGLang engine, bf16.**
   - Served in bf16, without `--quantization fp8`, as the 26B-A4B (for which two FP8 runs of the same questions agreed
     on only 95.7% of the decisions, against 99.35% in bf16). We did not test FP8 at load for this model.
   - There is no `--mamba-radix-cache-strategy` flag (Gemma 4 has no recurrent layers).
   - SGLang reserves `--mem-fraction-static` (default 0.85) of the GPU's memory for the weights and its KV cache: about
     82 GB on a 96 GB card, of which the weights take about 63 GB. The value only sizes the KV cache; scores do not
     depend on it. Every run below used one 96 GB H20 per server; we have not run it on smaller GPUs.
   - SGLang treats the model as multimodal by itself. The Decision Index run sent text only.
2. **`/v1/systemone` front end:** `knowline_server.py`, `chat` style, temperature 1, 16 scoring threads, read twice up
   to 3,000 tokens, per-type temperatures from `temperatures.json` (score questions / 0.6944), box questions with
   `--box-prompt gemma --box-coords yxyx` (Gemma's `box_2d` format, y first) and `box_calibration.json` when present.

```bash
pip install "sglang==0.5.21" "transformers==5.12.1" requests   # torch / CUDA per SGLang's install docs
bash serve_knowline.sh PelaAI/KnowLine-31B-Gen1 0 8080            # GPU 0; SGLang on :9080, /v1/systemone on :8080
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "m",
  "state": "Customer: my order arrived broken, I want my money back.",
  "questions": {"refund": {"type": "noul", "instructions": "Should the agent offer a refund?"},
                "tone": {"type": "choice", "instructions": "Customer tone?",
                         "criteria": {"angry": "Angry", "neutral": "Neutral", "happy": "Happy"}}}}'
```

Equivalent manual launch, with the exact flags of our run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path PelaAI/KnowLine-31B-Gen1 --served-model-name m \
  --tp 1 --mem-fraction-static 0.85 --enable-fp32-lm-head --port 9080 &
python knowline_server.py --model PelaAI/KnowLine-31B-Gen1 --backend sglang --url http://127.0.0.1:9080 \
  --served-model-name m --temperature 1 --workers 16 --port 8080 --temperatures temperatures.json \
  --box-prompt gemma --box-coords yxyx          # read twice is on by default
```

Without SGLang (through transformers; text only, no prefix cache, bf16):

```bash
pip install torch "transformers==5.12.1" requests
python knowline_server.py --model PelaAI/KnowLine-31B-Gen1 --backend hf --port 8080 --temperatures temperatures.json
```

- Loaded in bf16, the weights take about 62.5 GB (58.3 GiB) of GPU memory.
- Checked on one H20, read twice on: on 30 random Decision Index 0.3 rows, 80 of the 82 answers were the same as in our
  SGLang run; the other two are near-ties (the top two options within 0.06 of each other in one of the two runs).
  Requests took 576.5 ms median one at a time.
- Images need the SGLang backend.

## Decision Index run

- Suite rows: the 140,620 rows that `decision-index run --edition 0.3` sends: `selected-rows.jsonl.gz`,
  `added-rows.jsonl.gz` and `gsm8k-rows.jsonl.gz` of `suite-0.3`, with ToolRet, BRIGHT and ACOS cut to their scoring
  subsets.
- Four identical servers ran on four GPUs, one each, with the settings above (`--mem-fraction-static 0.90`, which only
  sizes the KV cache). The rows are split round-robin into 16 shards, 4 per server, so that each server's prefix cache
  keeps the shared prefixes of its few concurrent requests. Each shard is a resumable `decision-index run` against its
  server.
- The run used the released `knowline_server.py` (read twice on by default) from a local copy of this repository before
  the upload. All 140,178 scoreable requests were answered; none was unsupported or truncated, and none failed.

```bash
decision-index run --edition 0.3 --engine http --option base_url=http://127.0.0.1:<port> --option model=m --no-verify \
  --compact --rows shards/<i>.jsonl.gz --out shards/<i>          # i = 0..63, run in parallel
cat shards/*/results.jsonl > results.jsonl
decision-index score --suite-dir suite-0.3 --results results.jsonl --engine http --out score
```

## Hardware and latency

- Four NVIDIA H20 (96 GB each), one server per GPU.
- **Latency:** we sent the same 300 random Decision Index 0.3 rows (477 questions) as for KnowLine-26B-A4B-Gen2 and Gen2.5,
  one at a time, to one server on one H20 (after a warm-up).
  - Requests took 340.3 ms median, 910.0 ms mean, 1,185.4 ms at the 80th percentile and 3,976.6 ms at the 95th.
  - 473 of the 477 answers were the same as in our Decision Index run; the other four are near-ties (the top two
    options within 0.06 of each other in one of the two runs). A second pass matched the first on 476 of 477.
  - KnowLine-26B-A4B-Gen2.5 on the same rows and hardware: 66.3 ms median, 151.1 ms mean and 628.6 ms at the 95th
    percentile.
- Latency was not measured on the board's reference hardware (1x RTX PRO 6000, latency-v1).

## One-command Decision Index run

`knowline_engine.py` (in this repository) is a Decision Index engine. It runs the same front end in process, so there
is no server to start by hand:

```bash
pip install "sglang==0.5.21" "transformers==5.12.1" requests        # plus the decision-index kit
git clone https://huggingface.co/PelaAI/KnowLine-31B-Gen1 && cd KnowLine-31B-Gen1   # puts the three files on the path
python -m decision_index pipeline --engine knowline_engine:KnowLine \
    --option model=PelaAI/KnowLine-31B-Gen1 --option revision=<commit> --out runs/KnowLine-31B-Gen1
```

- **Default backend and layout (the setting of our runs):** read twice is on; the engine starts SGLang with the flags
  above (bf16, `--mem-fraction-static 0.85`) on a free local port and stops it when the run ends. Use `--option gpu=<n>`
  to pick a GPU, and `--option sglang_python=<python>` if SGLang lives in another environment.
- **`--option backend=hf`:** transformers only, bf16, no server; slower, and not the setting of our runs.
- **Limits:** up to 64 questions per request and 255 options per question. Larger requests are reported as
  unsupported; nothing is truncated.
- **Temperature in both backends:** transformers and SGLang both read the final norm weight and
  `final_logit_softcapping` from these files, so both backends apply the folded temperature; the per-type temperatures
  of `temperatures.json` are applied by the front end, the same for both. Our Decision Index run and
  calibration numbers use SGLang.

## Acknowledgements

- Read-twice layout: FlexDM ([github.com/1000luca/FlexDM](https://github.com/1000luca/FlexDM), MIT licence).
- Input repetition: Xu et al., 2023, "Re-Reading Improves Reasoning in Large Language Models"
  ([arXiv:2309.06275](https://arxiv.org/abs/2309.06275)).
