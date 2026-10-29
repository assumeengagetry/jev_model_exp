---
license: other
task_categories:
- zero-shot-classification
- text-classification
task_ids:
- natural-language-inference
language:
- en
dataset_info:
- config_name: default
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: labels
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  - name: task
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 2404193236
    num_examples: 1899993
  - name: validation
    num_bytes: 31043593
    num_examples: 24971
  - name: test
    num_bytes: 31390848
    num_examples: 25440
  download_size: 1169409645
  dataset_size: 2466627677
- config_name: triplet
  features:
  - name: anchor
    dtype: string
  - name: positive
    dtype: string
  - name: negative
    dtype: string
  - name: task
    dtype: string
  - name: license
    dtype: string
  - name: license_use
    dtype: string
  splits:
  - name: train
    num_bytes: 2452199128
    num_examples: 1899993
  - name: validation
    num_bytes: 31669513
    num_examples: 24971
  - name: test
    num_bytes: 32024875
    num_examples: 25440
  download_size: 1172381982
  dataset_size: 2515893516
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
- config_name: triplet
  data_files:
  - split: train
    path: triplet/train-*
  - split: validation
    path: triplet/validation-*
  - split: test
    path: triplet/test-*
---

[tasksource](https://github.com/sileod/tasksource) classification tasks recasted as natural language inference. 
This dataset is intended to improve label understanding in [zero-shot classification HF pipelines](https://huggingface.co/docs/transformers/main/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
).

Inputs that are text pairs are separated by a newline (\n).
```python
from transformers import pipeline
classifier = pipeline(model="sileod/deberta-v3-base-tasksource-nli")
classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
```
[deberta-v3-base-tasksource-nli](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli)  now includes `label-nli` in its training mix (a relatively small portion, to keep the model general, but note that nli models work for label-like zero shot classification without specific supervision (https://aclanthology.org/D19-1404.pdf).


```
@article{sileo2023tasksource,
  title={tasksource: A Dataset Harmonization Framework for Streamlined NLP Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  year={2023}
}
```

Built with [build_label_nli.py](https://github.com/sileod/tasksource/blob/main/scripts/build_label_nli.py); [sources.yaml](sources.yaml) lists every task with its revision, license and row counts.
