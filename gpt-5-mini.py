import argparse
import base64
import json
import os
import re
from collections import defaultdict

import requests


OPENROUTER_API_KEY = ""
DATA_PATH = "C:/teja/Obvious-med/data/all_tasks.jsonl"
OUT_DIR = "C:/teja/Obvious-med/closed-source-results"
MODEL = "openai/gpt-5-mini"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_SAMPLES = 5
TIMEOUT = 600
TEMPERATURE = 0.0
TOP_P = 1.0

SYSTEM_PROMPT_BASE = (
    "You are a medical image triage assistant for MedObvious. "
    "Use only visible evidence. Do not guess or add explanations. "
    "If anomaly evidence is unclear, prefer normal/no-anomaly. "
    "Return only the final answer. "
    "Output format: MCQ -> one uppercase letter; binary -> yes or no; "
    "location -> one short label (top-left, top-right, top-center, middle-left, middle-center, middle-right, "
    "bottom-left, bottom-center, bottom-right, center, none, or rowX-colY)."
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
    m = re.search(r"\b(?:answer|option|choice)\s*[:\-]?\s*([A-Z])\b", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), raw
    m = re.search(r"^\s*([A-Z])\s*$", raw.upper())
    if m:
        return m.group(1), raw
    m = re.search(r"\b([A-Z])\b", raw.upper())
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


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_to_data_url(path: str) -> str:
    mime = guess_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_first_n_tasks(path: str, n: int):
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if len(tasks) == n:
                break
    return tasks


def _collect_text(node):
    out = []
    if isinstance(node, str):
        t = node.strip()
        if t:
            out.append(t)
        return out
    if isinstance(node, list):
        for item in node:
            out.extend(_collect_text(item))
        return out
    if isinstance(node, dict):
        for key in ("text", "output_text", "content", "value", "message", "summary", "parts", "items"):
            if key in node:
                out.extend(_collect_text(node.get(key)))
    return out


def extract_response_parts(response_json):
    choices = response_json.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return "", None

    choice0 = choices[0]
    message = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}

    content = ""
    for node in (message.get("content"), choice0.get("text"), response_json.get("output_text"), response_json.get("output")):
        texts = _collect_text(node)
        if texts:
            content = "\n".join(texts).strip()
            break

    reasoning = None
    for node in (message.get("reasoning"), choice0.get("reasoning")):
        texts = _collect_text(node)
        if texts:
            reasoning = "\n".join(texts).strip()
            break

    return content, reasoning


def query_openrouter(question: str, image_path: str, task_type: str, max_tokens, use_system_prompt: bool):
    data_url = image_to_data_url(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    if use_system_prompt:
        messages.insert(0, {"role": "system", "content": build_system_prompt(task_type)})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        url=BASE_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=TIMEOUT,
    )
    return response


def pct(correct: int, total: int) -> float:
    return (correct / total * 100.0) if total else 0.0


def metric_row(correct: int, total: int):
    return {"correct": correct, "total": total, "accuracy": round(pct(correct, total), 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="How many samples to load from all_tasks.jsonl.")
    parser.add_argument("--use_system_prompt", action="store_true", help="Include system prompt (disabled by default).")
    parser.add_argument("--use_task_max_tokens", action="store_true", help="Use per-task max_tokens caps. Disabled by default.")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("Set OPENROUTER_API_KEY in final-gpt5.py before running.")

    tasks = load_first_n_tasks(DATA_PATH, args.max_samples)
    if not tasks:
        print("No tasks selected. Check DATA_PATH.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    n_suffix = str(args.max_samples)
    out_path = os.path.join(OUT_DIR, f"final-gpt5_preds_n{n_suffix}.jsonl")
    metrics_path = os.path.join(OUT_DIR, f"final-gpt5-metric-values-n{n_suffix}.json")
    metrics_latest_path = os.path.join(OUT_DIR, "final-gpt5-metric-values.json")

    per_version_selected = defaultdict(int)
    for task in tasks:
        task_id = task.get("id", "")
        version = task.get("version") or (task_id.split("_", 1)[0] if "_" in task_id else "unknown")
        per_version_selected[version] += 1

    print(f"Loaded {len(tasks)} tasks from {DATA_PATH}")
    print(f"Model: {MODEL}")
    print("Per-version counts:", dict(per_version_selected))
    print(f"Writing predictions to: {out_path}")

    total = 0
    correct = 0
    per_version = defaultdict(lambda: {"n": 0, "correct": 0})
    per_type = defaultdict(lambda: {"n": 0, "correct": 0})
    per_is_negative = defaultdict(lambda: {"n": 0, "correct": 0})

    with open(out_path, "w", encoding="utf-8") as wf:
        for i, task in enumerate(tasks, start=1):
            task_id = task.get("id", "")
            version = task.get("version") or (task_id.split("_", 1)[0] if "_" in task_id else "unknown")
            task_type = task.get("task_type", "unknown")
            gold = task.get("answer", None)
            img_path = task.get("image_path", None)
            question = task.get("question", "")
            is_negative_raw = task.get("is_negative", None)
            is_negative = normalize_is_negative(is_negative_raw)

            if is_negative is True:
                polarity_bucket = "negative"
            elif is_negative is False:
                polarity_bucket = "positive"
            else:
                polarity_bucket = "unknown"

            pred = None
            pred_raw = ""
            reasoning = None
            err = None

            if not img_path or not os.path.exists(img_path):
                err = f"missing_image: {img_path}"
            else:
                try:
                    max_tokens = MAX_TOKENS_BY_TYPE.get(task_type, 96) if args.use_task_max_tokens else None
                    response = query_openrouter(
                        question=question,
                        image_path=img_path,
                        task_type=task_type,
                        max_tokens=max_tokens,
                        use_system_prompt=args.use_system_prompt,
                    )
                    response.raise_for_status()

                    response_json = response.json()
                    pred_raw, reasoning = extract_response_parts(response_json)

                    if not (pred_raw or "").strip():
                        retry_response = query_openrouter(
                            question=question,
                            image_path=img_path,
                            task_type=task_type,
                            max_tokens=None,
                            use_system_prompt=False,
                        )
                        retry_response.raise_for_status()
                        retry_json = retry_response.json()
                        retry_pred_raw, retry_reasoning = extract_response_parts(retry_json)
                        if (retry_pred_raw or "").strip():
                            pred_raw = retry_pred_raw
                            reasoning = retry_reasoning

                    if task_type.endswith("_mcq"):
                        pred, _ = parse_mcq_letter(pred_raw)
                    elif task_type == "visual_referring":
                        pred, _ = parse_yes_no(pred_raw)
                    elif task_type.endswith("_open"):
                        pred, _ = parse_position(pred_raw)
                except Exception as ex:
                    err = f"api_error: {repr(ex)}"

            if err is None and pred is None:
                clean_raw = (pred_raw or "").strip()
                err = f"unparsed_response: {clean_raw[:240]}" if clean_raw else "empty_response_or_unrecognized_schema"

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
                "model": MODEL,
                "image_path": img_path,
                "is_negative": is_negative,
                "is_negative_raw": is_negative_raw,
                "gold": gold,
                "pred": pred,
                "final_pred_answer": pred,
                "pred_raw": pred_raw.strip() if isinstance(pred_raw, str) else pred_raw,
                "reasoning": reasoning,
                "correct": bool(is_correct),
                "error": err,
            }
            wf.write(json.dumps(row) + "\n")

            if i % 25 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] running acc: {pct(correct, total):.2f}%")

    metrics = {
        "prediction_file": out_path,
        "model": MODEL,
        "overall": metric_row(correct, total),
        "per_version": {},
        "per_task_type": {},
        "by_is_negative": {},
    }

    for version in sorted(per_version.keys()):
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
    print(f"Model: {MODEL}")
    print(f"Output: {out_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Overall: {correct}/{total} = {pct(correct, total):.2f}%")

    print("\nPer version:")
    for version in sorted(per_version.keys()):
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
