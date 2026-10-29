"""Capture actual SDPA K/V shapes and GQA dispatch for identical padded tokens."""
import hashlib
import json
import os
import sys

from benchmark import ROOT, PILOT, R0, REFERENCE, now, write


def main():
    import torch
    from transformers import DataCollatorWithPadding
    os.environ["LU_RESEARCH_ROOT"] = str(REFERENCE)
    sys.path.insert(0, str(PILOT / "code"))
    from model_utils import load_model
    torch.set_num_threads(8)
    model, tokenizer = load_model(R0 / "checkpoints/final", mode="causal", device=0, trainable=True)
    model.gradient_checkpointing_disable()
    model.train()
    rows = json.loads((ROOT / "inputs-medium.json").read_text())[:2]
    batch = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=256, pad_to_multiple_of=8)(rows)
    batch.pop("labels")
    batch = {key: value.cuda() for key, value in batch.items()}
    original_mask = batch["attention_mask"].clone()
    original_function = torch.nn.functional.scaled_dot_product_attention
    captured = []
    current = {}

    def wrapped(query, key, value, *args, **kwargs):
        mask = kwargs.get("attn_mask")
        captured.append({"control": current["name"], "q_shape": list(query.shape), "k_shape": list(key.shape),
                         "v_shape": list(value.shape), "mask_shape": list(mask.shape) if mask is not None else None,
                         "mask_dtype": str(mask.dtype) if mask is not None else None,
                         "enable_gqa": kwargs.get("enable_gqa", False), "is_causal": kwargs.get("is_causal")})
        return original_function(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = wrapped
    try:
        for name in ("real_padding_mask", "all_ones_control"):
            current["name"] = name
            batch["attention_mask"].copy_(original_mask)
            if name == "all_ones_control":
                batch["attention_mask"].fill_(1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch)
        torch.cuda.synchronize()
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_function
    unique = {json.dumps(item, sort_keys=True) for item in captured}
    report = {"state": "complete", "time": now(), "input_shape": list(batch["input_ids"].shape),
              "same_input_ids_both_controls": True, "layer_calls_per_control": 28,
              "captured": [json.loads(item) for item in sorted(unique)], "optimizer_updates": 0}
    assert len(captured) == 56
    masked = next(item for item in report["captured"] if item["control"] == "real_padding_mask")
    dense = next(item for item in report["captured"] if item["control"] == "all_ones_control")
    assert masked["k_shape"][1] == 16 and not masked["enable_gqa"] and masked["mask_shape"] is not None
    assert dense["k_shape"][1] == 8 and dense["enable_gqa"] and dense["mask_shape"] is None
    write(ROOT / "attention-dispatch.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
