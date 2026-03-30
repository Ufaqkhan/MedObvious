import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


DATA_PATH = "/home/ubuntu/MedObvious/data/all_tasks.jsonl"
OUT_DIR = "/home/ubuntu/MedObvious/results"
MODEL = "MBZUAI/BiMediX2-8B"
TEMPERATURE = 0.0
TOP_P = 1.0

SYSTEM_PROMPT_BASE = (
    "You are a medical image reasoning assistant for MedObvious, a pre-diagnostic visual triage benchmark. "
    "Focus on basic visual sanity checks before diagnosis: orientation, body-part and modality consistency, image integrity, and obvious artifacts. "
    "Do not invent findings. If evidence of anomaly is not visible, prefer the normal or no-anomaly option. "
    "Return only the final answer with no explanation. "
    "Formatting rules: for multiple-choice output one uppercase letter A-E; for binary output exactly yes or no; "
    "for location output one short label such as top-left, top-right, top-center, middle-left, middle-center, middle-right, "
    "bottom-left, bottom-center, bottom-right, center, none, or rowX-colY."
)

MAX_TOKENS_BY_TYPE = {
    "detection_mcq": 32,
    "referring_mcq": 32,
    "detection_open": 96,
    "referring_open": 96,
    "visual_referring": 64,
}


def build_system_prompt(task_type: str) -> str:
    return f"{SYSTEM_PROMPT_BASE} Task type: {task_type}."


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", "-")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_position(s: str) -> str:
    s = normalize_text(s)
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


def normalize_is_negative(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        t = normalize_text(value)
        if t in {"true", "1", "yes", "y", "negative"}:
            return True
        if t in {"false", "0", "no", "n", "positive"}:
            return False
    return None


def parse_mcq_letter(text: str):
    raw = (text or "").strip()
    m = re.search(r"\b([A-E])\b", raw.upper())
    if m:
        return m.group(1), raw
    m = re.search(r"([A-E])", raw.upper())
    if m:
        return m.group(1), raw
    return None, raw


def parse_yes_no(text: str):
    raw = (text or "").strip()
    t = normalize_text(raw)
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1), raw
    return None, raw


def parse_position(text: str):
    raw = (text or "").strip()
    t = normalize_position(raw)
    candidates = [
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
        "top-center",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-center",
        "center",
        "centre",
        "none",
        "no-outlier",
        "all-same",
        "all-the-same",
    ]
    for candidate in candidates:
        if candidate in t:
            if candidate == "centre":
                return "center", raw
            return candidate, raw
    m = re.search(r"\brow\s*([1-9])\s*(?:,|\s)\s*(?:col|column)\s*([1-9])\b", normalize_text(raw))
    if m:
        return f"row{m.group(1)}-col{m.group(2)}", raw
    return None, raw


def is_correct_for_task(task_type: str, pred, pred_raw: str, gold):
    if gold is None:
        return False
    if task_type.endswith("_mcq"):
        return pred == str(gold).strip().upper()
    if task_type == "visual_referring":
        return (pred or "").lower() == str(gold).strip().lower()
    if task_type.endswith("_open"):
        gold_norm = normalize_position(str(gold))
        out_norm = normalize_position(pred_raw)
        if gold_norm and gold_norm in out_norm:
            return True
        if pred is not None:
            return normalize_position(str(pred)) == gold_norm
        return False
    return False


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_torch_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_components():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    except Exception:
        processor = None

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": pick_torch_dtype(),
    }

    model = None
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", **model_kwargs)
        except Exception:
            try:
                model = AutoModel.from_pretrained(MODEL, device_map="auto", **model_kwargs)
            except Exception:
                model = None

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
        except Exception:
            model = AutoModel.from_pretrained(MODEL, **model_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.eval()
    return model, tokenizer, processor


def first_model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_prompt(system_prompt: str, question: str, processor, tokenizer):
    messages_variants = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>\n{question}"},
        ],
    ]

    for messages in messages_variants:
        try:
            if processor is not None and hasattr(processor, "apply_chat_template"):
                return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"{system_prompt}\n\n{question}\nAnswer:"


def decode_generated(model_inputs, generated_ids, tokenizer, processor):
    if "input_ids" in model_inputs:
        prompt_len = model_inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[:, prompt_len:]

    if processor is not None and hasattr(processor, "batch_decode"):
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


def chat_completion_local(image_path: str, question: str, task_type: str, max_tokens: int, model, tokenizer, processor):
    image = Image.open(image_path).convert("RGB")
    system_prompt = build_system_prompt(task_type)
    prompt = prepare_prompt(system_prompt, question, processor, tokenizer)

    if processor is not None:
        model_inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        model_inputs = tokenizer(prompt, return_tensors="pt")

    device = first_model_device(model)
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    return decode_generated(model_inputs, generated_ids, tokenizer, processor)


def pct(correct: int, total: int) -> float:
    return (correct / total * 100.0) if total else 0.0


def metric_row(correct: int, total: int):
    return {"correct": correct, "total": total, "accuracy": round(pct(correct, total), 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e",
        type=int,
        default=0,
        help="How many samples to take from each version. 0 means all.",
    )
    args = parser.parse_args()
    e = args.e

    os.makedirs(OUT_DIR, exist_ok=True)
    e_suffix = f"{e}" if e > 0 else "all"
    out_path = os.path.join(OUT_DIR, f"bimedi-8B_preds_e{e_suffix}.jsonl")
    metrics_path = os.path.join(OUT_DIR, f"bimedi-8B-metric-values-e{e_suffix}.json")
    metrics_latest_path = os.path.join(OUT_DIR, "bimedi-8B-metric-values.json")

    buckets = OrderedDict()
    counts_seen = defaultdict(int)

    for task in load_jsonl(DATA_PATH):
        task_id = task.get("id", "")
        version = task_id.split("_", 1)[0] if "_" in task_id else "unknown"
        if version not in buckets:
            buckets[version] = []
        if e > 0 and counts_seen[version] >= e:
            continue
        buckets[version].append(task)
        counts_seen[version] += 1

    tasks = []
    for version in buckets.keys():
        tasks.extend(buckets[version])

    if not tasks:
        print("No tasks selected. Check DATA_PATH.")
        return

    print(f"Loaded {len(tasks)} tasks from {DATA_PATH}")
    print("Per-version counts:", {v: len(ts) for v, ts in buckets.items()})
    print(f"Writing predictions to: {out_path}")
    print(f"Loading model from HF: {MODEL}")

    model, tokenizer, processor = load_model_components()

    total = 0
    correct = 0
    per_version = defaultdict(lambda: {"n": 0, "correct": 0})
    per_type = defaultdict(lambda: {"n": 0, "correct": 0})
    per_is_negative = defaultdict(lambda: {"n": 0, "correct": 0})

    with open(out_path, "w", encoding="utf-8") as wf:
        for i, task in enumerate(tasks, start=1):
            task_id = task.get("id", "")
            version = task_id.split("_", 1)[0] if "_" in task_id else "unknown"
            task_type = task.get("task_type", "unknown")
            question = task.get("question", "")
            gold = task.get("answer", None)
            img_path = task.get("image_path", None)
            is_negative_raw = task.get("is_negative", None)
            is_negative = normalize_is_negative(is_negative_raw)

            if is_negative is True:
                polarity_bucket = "negative"
            elif is_negative is False:
                polarity_bucket = "positive"
            else:
                polarity_bucket = "unknown"

            max_tokens = MAX_TOKENS_BY_TYPE.get(task_type, 96)
            pred = None
            pred_raw = ""
            err = None

            if not img_path or not os.path.exists(img_path):
                err = f"missing_image: {img_path}"
            else:
                try:
                    pred_raw = chat_completion_local(
                        img_path,
                        question,
                        task_type=task_type,
                        max_tokens=max_tokens,
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                    )
                    if task_type.endswith("_mcq"):
                        pred, _ = parse_mcq_letter(pred_raw)
                    elif task_type == "visual_referring":
                        pred, _ = parse_yes_no(pred_raw)
                    elif task_type.endswith("_open"):
                        pred, _ = parse_position(pred_raw)
                    else:
                        pred = None
                except Exception as ex:
                    err = f"api_error: {repr(ex)}"

            is_correct = is_correct_for_task(task_type, pred, pred_raw, gold)

            total += 1
            correct += int(is_correct)
            per_version[version]["n"] += 1
            per_version[version]["correct"] += int(is_correct)
            per_type[task_type]["n"] += 1
            per_type[task_type]["correct"] += int(is_correct)
            per_is_negative[polarity_bucket]["n"] += 1
            per_is_negative[polarity_bucket]["correct"] += int(is_correct)

            row = {
                "id": task_id,
                "version": version,
                "task_type": task_type,
                "image_path": img_path,
                "is_negative": is_negative,
                "is_negative_raw": is_negative_raw,
                "gold": gold,
                "pred": pred,
                "pred_raw": pred_raw.strip() if isinstance(pred_raw, str) else pred_raw,
                "correct": bool(is_correct),
                "error": err,
            }
            wf.write(json.dumps(row) + "\n")

            if i % 25 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] running acc: {pct(correct, total):.2f}%")

    metrics = {
        "prediction_file": out_path,
        "overall": metric_row(correct, total),
        "per_version": {},
        "per_task_type": {},
        "by_is_negative": {},
    }

    for version in buckets.keys():
        n = per_version[version]["n"]
        c = per_version[version]["correct"]
        metrics["per_version"][version] = metric_row(c, n)

    for task_type in sorted(per_type.keys()):
        n = per_type[task_type]["n"]
        c = per_type[task_type]["correct"]
        metrics["per_task_type"][task_type] = metric_row(c, n)

    for bucket in ["negative", "positive", "unknown"]:
        n = per_is_negative[bucket]["n"]
        c = per_is_negative[bucket]["correct"]
        metrics["by_is_negative"][bucket] = metric_row(c, n)

    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    with open(metrics_latest_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Output: {out_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Overall: {correct}/{total} = {pct(correct, total):.2f}%")

    print("\nPer version:")
    for version in buckets.keys():
        n = per_version[version]["n"]
        c = per_version[version]["correct"]
        print(f"  {version}: {c}/{n} = {pct(c, n):.2f}%")

    print("\nPer task_type:")
    for task_type in sorted(per_type.keys()):
        n = per_type[task_type]["n"]
        c = per_type[task_type]["correct"]
        print(f"  {task_type}: {c}/{n} = {pct(c, n):.2f}%")

    print("\nBy is_negative:")
    for bucket in ["negative", "positive", "unknown"]:
        n = per_is_negative[bucket]["n"]
        c = per_is_negative[bucket]["correct"]
        print(f"  {bucket}: {c}/{n} = {pct(c, n):.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
