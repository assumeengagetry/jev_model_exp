
# JevEmbed-Data
![JevEmbed pixel-art banner showing embeddings leading to Choice, Score, and Noul decisions](assets/jevembed-banner.png)

This dataset contains **1,601,157 training** and **66,482 test** questions for [JevEmbed](https://github.com/HITsz-TMG/JevEmbed). Here, `test` is the source corpus's validation split. The ten Parquet files each hold at most 200,000 rows.

## Fields

Each row is one decision. `request_json` and `answers_json` are JSON strings; parse them with `json.loads`.

| Field | Meaning |
|---|---|
| `id` | Unique question ID. |
| `group` | Case ID. Related questions stay in the same split. |
| `request_json` | Input: `state` plus a `decision` question with its type, instructions, and optional criteria. |
| `answers_json` | Reference answer for `decision`. This is a training label, not model input. |
| `question_type` | `choice`, `score`, or `noul`. Also present inside `request_json`. |
| `source` | Short source name. |
| `source_repo` | Source dataset or project generator. |
| `source_revision` | Source version. |
| `source_row` | Row ID in that source. |
| `domain` | Topic tag for filtering or sampling. |
| `license` | License recorded for that source. |
| `original_split` | `train` or `validation`; the latter is published here as `test`. |

Inside `request_json`, `state` is the case to judge. `questions.decision` has the task type, instructions, and any candidate descriptions (`criteria`). `answers_json` has the matching label. For example, a Choice answer can name one option or give a probability for each option. Score uses ordered levels; Noul gives a yes probability between 0 and 1. Hard and soft labels are kept.

## Use with JevEmbed

JevEmbed uses one encoder for the question and its candidates. It compares their embeddings, then learns from the answer:

| Task | What the encoder compares | Training target |
|---|---|---|
| Choice | State and instruction against each named option. | Chosen option or option probabilities. |
| Score | State and instruction against ordered score descriptions. | Chosen level, level probabilities, or expected score. |
| Noul | State against the question and, when given, its true/false descriptions. | Yes probability. |

Choice and level-based Score use cosine similarity and a softmax loss. A numeric Score uses the expected level and squared error. Noul uses a yes/no loss on cosine similarity or the difference between true and false similarities. The answer and source fields are not put into the encoder input.

The current JevEmbed trainer reads JSONL. Convert the Parquet rows before training:

```python
import json
from datasets import load_dataset

data = load_dataset("YOUR_NAMESPACE/JevEmbed-Data", streaming=True)
for split, path in (("train", "train.jsonl"), ("test", "validation.jsonl")):
    with open(path, "w", encoding="utf-8") as out:
        for row in data[split]:
            record = dict(id=row["id"], group=row["group"],
                          request=json.loads(row["request_json"]),
                          answers=json.loads(row["answers_json"]))
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
```

In the JevEmbed repository, copy `configs/training/lora.yaml` to `lora-2048.yaml`, set `max_input_tokens: 2048`, and train KaLM v2.5 with:

```bash
python -m jevembed.training \
  --config configs/kalm-embedding-v2.5.yaml \
  --training-config ./lora-2048.yaml \
  --train-data train.jsonl \
  --eval-data validation.jsonl \
  --output artifacts/jevembed-data-kalm
```

See the [training guide](https://github.com/HITsz-TMG/JevEmbed/blob/main/docs/training.md) for setup and other models. Input lengths were checked with the KaLM tokenizer; check them again for a different model.

See [sources and checks](docs/PROCESSING_REPORT.md). This dataset has **mixed licenses**; check each row's `license` and the source report before reuse or redistribution.

## Citation

If you find JevEmbed-Data useful, please consider citing the following papers:

```bibtex
@misc{zhao2025kalmembeddingv2,
      title={KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model},
      author={Xinping Zhao and Xinshuo Hu and Zifei Shan and Shouzheng Huang and Yao Zhou and Xin Zhang and Zetian Sun and Zhenyu Liu and Dongfang Li and Xinyuan Wei and Youcheng Pan and Yang Xiang and Meishan Zhang and Haofen Wang and Jun Yu and Baotian Hu and Min Zhang},
      year={2025},
      eprint={2506.20923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.20923},
}

@misc{hu2025kalmembedding,
      title={KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model},
      author={Xinshuo Hu and Zifei Shan and Xinping Zhao and Zetian Sun and Zhenyu Liu and Dongfang Li and Shaolin Ye and Xinyuan Wei and Qian Chen and Baotian Hu and Haofen Wang and Jun Yu and Min Zhang},
      year={2025},
      eprint={2501.01028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.01028},
}
```