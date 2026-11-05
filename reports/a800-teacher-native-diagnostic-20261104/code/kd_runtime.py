"""Raw pretrained Qwen backbone with the same saved, never-trained adapter/head."""
import json
import os
from pathlib import Path
import sys

import numpy as np

SOURCE_ROOT = Path("/data/run01/scwb721/jev-kd-20261101")
ROOT = Path(os.environ["JEV_KD_ROOT"])
WORKSPACE = SOURCE_ROOT / "workspace"
POOL = WORKSPACE / "cache/pair-training/global-pool"
DOMAINS = ["knowledge", "retrieval", "tools", "arts"]
PARENT = WORKSPACE / "models/Qwen3-0.6B-common-pair-init-r32"
sys.path.insert(0, str(WORKSPACE / "scripts"))
import pair_training_runtime as legacy
legacy.DOMAINS = DOMAINS


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class PilotPool:
    def __init__(self):
        self.indices = np.load(ROOT / "selection/train_indices.npy", mmap_mode="r")
        self.full = legacy.PairPool("MIXED_ALL", root=POOL)
        self.size = len(self.indices)
        self.domains = DOMAINS

    def __len__(self):
        return self.size

    def fetch(self, indices):
        canonical = self.indices[np.asarray(indices, dtype=np.int64)]
        return self.full.fetch(canonical)

    order = legacy.PairPool.order


def pool():
    if (ROOT / "scope.json").is_file():
        return legacy.PairPool("MIXED_ALL", root=POOL)
    return PilotPool()


class TokenStore:
    def __init__(self):
        manifest = json.loads((POOL / "MANIFEST.json").read_text())
        self.boundaries = {"prefix": manifest["typed_prefix_boundary"], "suffix": manifest["typed_suffix_boundary"]}
        self.data = {}
        self.lengths = {}
        folders = [WORKSPACE / "cache/pair-training/tokenized" / name for name in ["typed-full", "global-new"]]
        for role in ["prefix", "suffix"]:
            streams = []
            lengths = []
            for folder in folders:
                ids = np.memmap(folder / f"{role}_ids.bin", mode="r", dtype=np.int32)
                offsets = np.memmap(folder / f"{role}_offsets.bin", mode="r", dtype=np.uint64)
                streams.append((ids, offsets))
                lengths.append(np.diff(offsets).astype(np.uint32))
            self.data[role] = streams
            self.lengths[role] = np.concatenate(lengths)

    get = legacy.TokenStore.get


def shared_order(dataset):
    path = ROOT / "outputs/shared-order.npy"
    if not path.exists():
        temporary = path.with_suffix(".writing.npy")
        np.save(temporary, dataset.order(42, batch_size=128))
        temporary.replace(path)
    order = np.load(path, mmap_mode="r")
    if len(order) != len(dataset):
        raise RuntimeError("Existing pilot schedule belongs to a different sample")
    return order


def load_classifier(checkpoint=PARENT, trainable=False, attention="auto"):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    from safetensors import safe_open

    if attention == "auto":
        try:
            import flash_attn
            attention = "flash_attention_2"
        except ImportError:
            attention = "sdpa"
    checkpoint = Path(checkpoint)
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError(f"Raw initialization or owned resume checkpoint is absent: {checkpoint}")
    adapter = json.loads((checkpoint / "adapter_config.json").read_text())
    if adapter.get("r") != 32 or adapter.get("lora_alpha") != 64 or adapter.get("task_type") != "SEQ_CLS":
        raise RuntimeError("Checkpoint must use the shared r32/alpha64 sequence-classifier architecture")
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        if not any("score" in key and key.endswith("weight") for key in tensors.keys()):
            raise RuntimeError("The common saved score head is absent; refusing a different random head")
    tokenizer = AutoTokenizer.from_pretrained(WORKSPACE / "models/Qwen3-0.6B-common-pair-init-r32", local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.num_labels = 2
    config.id2label = {0: "not_match", 1: "match"}
    config.label2id = {"not_match": 0, "match": 1}
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    base = AutoModelForSequenceClassification.from_pretrained(
        WORKSPACE / "models/Qwen3-0.6B", config=config, torch_dtype=torch.bfloat16,
        attn_implementation=attention, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=trainable, local_files_only=True)
    model.config.use_cache = False
    model.to("cuda:0")
    model.train(trainable)
    return model, tokenizer
