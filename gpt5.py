import base64
import json
import os
import re

import requests


OPENROUTER_API_KEY = ""
DATA_PATH = "C:/teja/Obvious-med/data/all_tasks.jsonl"
MODEL = "openai/gpt-5-nano"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_SAMPLES = 5
TIMEOUT = 600


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


def query_openrouter(question: str, image_path: str):
    data_url = image_to_data_url(image_path)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

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


def main():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("Set OPENROUTER_API_KEY in this file before running.")

    tasks = load_first_n_tasks(DATA_PATH, MAX_SAMPLES)
    print(json.dumps({"data_path": DATA_PATH, "loaded_samples": len(tasks), "max_samples": MAX_SAMPLES}))

    for i, task in enumerate(tasks, start=1):
        task_id = task.get("id", "")
        version = task.get("version") or (task_id.split("_", 1)[0] if "_" in task_id else "unknown")
        task_type = task.get("task_type", "unknown")
        gold = task.get("answer", None)
        is_negative_raw = task.get("is_negative", None)
        is_negative = normalize_is_negative(is_negative_raw)

        row = {
            "id": task_id,
            "version": version,
            "task_type": task_type,
            "model": MODEL,
            "image_path": task.get("image_path"),
            "is_negative": is_negative,
            "is_negative_raw": is_negative_raw,
            "gold": gold,
            "pred": None,
            "final_pred_answer": None,
            "pred_raw": "",
            "reasoning": None,
            "correct": False,
            "error": None,
        }

        image_path = task.get("image_path")
        question = task.get("question", "")
        if not image_path or not os.path.exists(image_path):
            row["error"] = f"missing_image: {image_path}"
            print(json.dumps(row, ensure_ascii=False))
            continue

        try:
            response = query_openrouter(question=question, image_path=image_path)
            response.raise_for_status()

            response_json = response.json()
            pred_raw, reasoning = extract_response_parts(response_json)
            row["pred_raw"] = pred_raw
            row["reasoning"] = reasoning

            pred = None
            if task_type.endswith("_mcq"):
                pred, _ = parse_mcq_letter(pred_raw)
            elif task_type == "visual_referring":
                pred, _ = parse_yes_no(pred_raw)
            elif task_type.endswith("_open"):
                pred, _ = parse_position(pred_raw)

            row["pred"] = pred
            row["final_pred_answer"] = pred
            row["correct"] = bool(is_correct_for_task(task_type, pred, pred_raw, gold))

            if pred is None:
                clean_raw = (pred_raw or "").strip()
                row["error"] = f"unparsed_response: {clean_raw[:240]}" if clean_raw else "empty_response_or_unrecognized_schema"
        except Exception as ex:
            row["error"] = repr(ex)

        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()



























# import base64
# import json
# import os

# import requests


# OPENROUTER_API_KEY = ""
# DATA_PATH = "C:/teja/Obvious-med/data/all_tasks.jsonl"
# MODEL = "openai/gpt-5-nano"
# BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# MAX_SAMPLES = 5
# TIMEOUT = 600


# def guess_mime(path: str) -> str:
#     ext = os.path.splitext(path)[1].lower()
#     if ext in [".jpg", ".jpeg"]:
#         return "image/jpeg"
#     if ext == ".png":
#         return "image/png"
#     if ext == ".webp":
#         return "image/webp"
#     return "application/octet-stream"


# def image_to_data_url(path: str) -> str:
#     mime = guess_mime(path)
#     with open(path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("utf-8")
#     return f"data:{mime};base64,{b64}"


# def load_first_n_tasks(path: str, n: int):
#     tasks = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             tasks.append(json.loads(line))
#             if len(tasks) == n:
#                 break
#     return tasks


# def query_openrouter(question: str, image_path: str):
#     data_url = image_to_data_url(image_path)
#     payload = {
#         "model": MODEL,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                     {"type": "image_url", "image_url": {"url": data_url}},
#                 ],
#             }
#         ],
#     }

#     response = requests.post(
#         url=BASE_URL,
#         headers={
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#         },
#         data=json.dumps(payload),
#         timeout=TIMEOUT,
#     )
#     return response


# def main():
#     if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
#         raise ValueError("Set OPENROUTER_API_KEY in this file before running.")

#     tasks = load_first_n_tasks(DATA_PATH, MAX_SAMPLES)
#     print(json.dumps({"data_path": DATA_PATH, "loaded_samples": len(tasks), "max_samples": MAX_SAMPLES}))

#     for i, task in enumerate(tasks, start=1):
#         row = {
#             "idx": i,
#             "id": task.get("id"),
#             "version": task.get("version"),
#             "task_type": task.get("task_type"),
#             "model": MODEL,
#             "image_path": task.get("image_path"),
#         }

#         image_path = task.get("image_path")
#         question = task.get("question", "")
#         if not image_path or not os.path.exists(image_path):
#             row["error"] = f"missing_image: {image_path}"
#             print(json.dumps(row, ensure_ascii=False))
#             continue

#         try:
#             response = query_openrouter(question=question, image_path=image_path)
#             row["status_code"] = response.status_code
#             row["raw_response"] = response.text
#         except Exception as ex:
#             row["error"] = repr(ex)

#         print(json.dumps(row, ensure_ascii=False))


# if __name__ == "__main__":
#     main()
