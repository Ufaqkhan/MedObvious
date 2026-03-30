# #!/usr/bin/env python3
# import json
# import os
# import tempfile

# IN_PATH = "/home/umair/TW/Obvious-med/data/all_tasks.jsonl"

# OLD_PREFIX = "/share_2/users/umair_nawaz/MedObvious/data/benchmark_refined/"
# NEW_PREFIX = "/home/umair/TW/Obvious-med/data/"


# def main() -> None:
#     if not os.path.isfile(IN_PATH):
#         raise FileNotFoundError(f"Input file not found: {IN_PATH}")

#     in_dir = os.path.dirname(IN_PATH) or "."
#     changed = 0
#     total = 0

#     # Write to a temp file in the same directory, then atomically replace original.
#     fd, tmp_path = tempfile.mkstemp(prefix=".all_tasks.", suffix=".jsonl.tmp", dir=in_dir)
#     try:
#         with os.fdopen(fd, "w", encoding="utf-8") as out_f, open(IN_PATH, "r", encoding="utf-8") as in_f:
#             for line_no, line in enumerate(in_f, start=1):
#                 line = line.rstrip("\n")
#                 if not line.strip():
#                     continue  # skip empty lines

#                 try:
#                     obj = json.loads(line)
#                 except json.JSONDecodeError as e:
#                     raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

#                 total += 1
#                 ip = obj.get("image_path")
#                 if isinstance(ip, str) and ip.startswith(OLD_PREFIX):
#                     obj["image_path"] = NEW_PREFIX + ip[len(OLD_PREFIX):]
#                     changed += 1

#                 out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

#         os.replace(tmp_path, IN_PATH)
#     except Exception:
#         # Clean up temp file if something goes wrong
#         try:
#             os.remove(tmp_path)
#         except OSError:
#             pass
#         raise

#     print(f"Done. Processed {total} lines; updated {changed} image_path values.")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# import json
# import os
# import tempfile

# # Files to update (in-place)
# FILES = [
#     "/home/umair/TW/Obvious-med/data/all_tasks.jsonl",
#     "/home/umair/TW/Obvious-med/data/v1/tasks.jsonl",
#     "/home/umair/TW/Obvious-med/data/v2/tasks.jsonl",
#     "/home/umair/TW/Obvious-med/data/v3/tasks.jsonl",
#     "/home/umair/TW/Obvious-med/data/v4/tasks.jsonl",
#     "/home/umair/TW/Obvious-med/data/v5/tasks.jsonl",
# ]

# OLD_PREFIX = "/share_2/users/umair_nawaz/MedObvious/data/benchmark_refined/"
# NEW_PREFIX = "/home/umair/TW/Obvious-med/data/"


# def update_file(path: str) -> tuple[int, int]:
#     """Return (total_json_lines, changed_image_paths)."""
#     if not os.path.isfile(path):
#         raise FileNotFoundError(f"File not found: {path}")

#     out_dir = os.path.dirname(path) or "."
#     total = 0
#     changed = 0

#     fd, tmp_path = tempfile.mkstemp(prefix="._tmp_", suffix=".jsonl", dir=out_dir)
#     try:
#         with os.fdopen(fd, "w", encoding="utf-8") as out_f, open(path, "r", encoding="utf-8") as in_f:
#             for line_no, line in enumerate(in_f, start=1):
#                 raw = line.rstrip("\n")
#                 if not raw.strip():
#                     continue  # skip empty lines

#                 try:
#                     obj = json.loads(raw)
#                 except json.JSONDecodeError as e:
#                     raise ValueError(f"{path}: invalid JSON on line {line_no}: {e}") from e

#                 total += 1
#                 ip = obj.get("image_path")
#                 if isinstance(ip, str) and ip.startswith(OLD_PREFIX):
#                     obj["image_path"] = NEW_PREFIX + ip[len(OLD_PREFIX):]
#                     changed += 1

#                 out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

#         os.replace(tmp_path, path)
#     except Exception:
#         try:
#             os.remove(tmp_path)
#         except OSError:
#             pass
#         raise

#     return total, changed


# def main() -> None:
#     grand_total = 0
#     grand_changed = 0

#     for f in FILES:
#         total, changed = update_file(f)
#         grand_total += total
#         grand_changed += changed
#         print(f"{f}: processed {total} lines; updated {changed} image_path values.")

#     print(f"\nDone. Total processed {grand_total} lines; total updated {grand_changed} image_path values.")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
import json
import os
import sys

FILES = [
    "/home/umair/TW/Obvious-med/data/all_tasks.jsonl",
    "/home/umair/TW/Obvious-med/data/v1/tasks.jsonl",
    "/home/umair/TW/Obvious-med/data/v2/tasks.jsonl",
    "/home/umair/TW/Obvious-med/data/v3/tasks.jsonl",
    "/home/umair/TW/Obvious-med/data/v4/tasks.jsonl",
    "/home/umair/TW/Obvious-med/data/v5/tasks.jsonl",
]


def check_file(path: str) -> tuple[int, int, list[tuple[str, int, str, str]]]:
    """
    Returns:
      total_lines, missing_count, missing_samples[(file, line_no, id, image_path)]
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    total = 0
    missing = 0
    samples: list[tuple[str, int, str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {e}") from e

            total += 1
            img_path = obj.get("image_path")
            ex_id = obj.get("id", "<no-id>")

            if not isinstance(img_path, str) or not img_path:
                missing += 1
                if len(samples) < 25:
                    samples.append((path, line_no, str(ex_id), str(img_path)))
                continue

            if not os.path.exists(img_path):
                missing += 1
                if len(samples) < 25:
                    samples.append((path, line_no, str(ex_id), img_path))

    return total, missing, samples


def main() -> None:
    grand_total = 0
    grand_missing = 0
    all_samples: list[tuple[str, int, str, str]] = []

    for fp in FILES:
        total, missing, samples = check_file(fp)
        grand_total += total
        grand_missing += missing
        all_samples.extend(samples)

        print(f"{fp}: checked {total} rows; missing {missing} image paths.")

    print("\n=== Summary ===")
    print(f"Total rows checked: {grand_total}")
    print(f"Total missing paths: {grand_missing}")

    if grand_missing > 0:
        print("\n=== Missing examples (up to 25) ===")
        for f, line_no, ex_id, img_path in all_samples[:25]:
            print(f"- file={f} line={line_no} id={ex_id} image_path={img_path}")

        # Exit non-zero so it's easy to detect failure in scripts/CI
        sys.exit(2)

    print("All image paths exist ✅")


if __name__ == "__main__":
    main()