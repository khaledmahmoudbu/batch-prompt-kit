from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

MODEL = "gpt-5.4"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 2048

DEFAULT_INPUT_FILE_NAME = "data.jsonl"
DEFAULT_MAP_FILE_NAME = "map"
PROMPTS_DIR_NAME = "prompts"
BATCHES_DIR_NAME = "batches"
DEFINITION_DIR_NAME = "definition"
DEFINITION_FILE_NAME = "generated_batch.jsonl"
DEFINITION_MANIFEST_FILE_NAME = "generated_batch_manifest.jsonl"
BATCH_RESPONSES_FILE_NAME = "batch_responses.jsonl"


def task_root() -> Path:
    return Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", " ").replace("\x00", " ")
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
    return text.strip()


def read_prompt(batch_name: str) -> str:
    prompt_path = task_root() / PROMPTS_DIR_NAME / batch_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return [row for _, row in iter_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_field_map(dataset_folder: str) -> list[tuple[str, str]]:
    map_path = task_root() / dataset_folder / DEFAULT_MAP_FILE_NAME
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map file: {map_path}")

    mapping: list[tuple[str, str]] = []
    seen_prompt_fields: set[str] = set()

    with map_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid map entry at {map_path}:{line_no}. Expected ds_field=field_in_prompt")
            ds_field, prompt_field = line.split("=", 1)
            ds_field = ds_field.strip()
            prompt_field = prompt_field.strip()
            if not ds_field or not prompt_field:
                raise ValueError(f"Invalid map entry at {map_path}:{line_no}. Expected ds_field=field_in_prompt")
            if prompt_field in seen_prompt_fields:
                raise ValueError(f"Duplicate prompt field in map: {prompt_field}")
            mapping.append((ds_field, prompt_field))
            seen_prompt_fields.add(prompt_field)

    if not mapping:
        raise ValueError(f"Map file is empty: {map_path}")
    return mapping


def make_unique_id(batch_name: str, row: dict, source_line_no: int, seen: set[str]) -> str:
    source_hash = str(row.get("source_hash") or "").strip()
    sample_index = row.get("sample_index")

    if source_hash and sample_index is not None:
        candidate = f"{batch_name}-{source_hash}-{sample_index}"
    elif source_hash:
        candidate = f"{batch_name}-{source_hash}"
    else:
        candidate = f"{batch_name}-line-{source_line_no:08d}"

    if candidate not in seen:
        seen.add(candidate)
        return candidate

    suffix = 2
    while True:
        amended = f"{candidate}-{suffix}"
        if amended not in seen:
            seen.add(amended)
            return amended
        suffix += 1


def load_manifest_map(batch_name: str) -> dict[str, dict]:
    manifest_path = task_root() / BATCHES_DIR_NAME / batch_name / DEFINITION_DIR_NAME / DEFINITION_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    out: dict[str, dict] = {}
    for row in read_jsonl(manifest_path):
        unique_id = str(row.get("unique_id") or row.get("custom_id") or "").strip()
        if unique_id:
            out[unique_id] = row
    return out


def resolve_batch_responses_path(batch_name: str) -> Path:
    root = task_root()
    candidates = [
        root / BATCHES_DIR_NAME / batch_name / BATCH_RESPONSES_FILE_NAME,
        root / BATCHES_DIR_NAME / batch_name / "data.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing batch responses file. Looked for: " + ", ".join(str(p) for p in candidates)
    )


def build_user_payload(dataset_folder: str, row: dict, source_field_name: str | None) -> str:
    if source_field_name:
        value = normalize_text(str(row.get(source_field_name) or ""))
        if not value:
            raise ValueError(f"missing_or_empty_field:{source_field_name}")
        return value

    mapping = read_field_map(dataset_folder)
    payload: dict[str, str] = {}
    for ds_field, prompt_field in mapping:
        value = normalize_text(str(row.get(ds_field) or ""))
        if not value:
            raise ValueError(f"missing_or_empty_field:{ds_field}")
        payload[prompt_field] = value
    return json.dumps(payload, ensure_ascii=False)


def make_request_object(custom_id: str, developer_prompt: str, prompt_value: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": prompt_value},
            ],
        },
    }


def generate_batch(batch_name: str, dataset_folder: str, source_field_name: str | None = None) -> None:
    root = task_root()
    input_path = root / dataset_folder / DEFAULT_INPUT_FILE_NAME
    definition_dir = root / BATCHES_DIR_NAME / batch_name / DEFINITION_DIR_NAME
    definition_file = definition_dir / DEFINITION_FILE_NAME
    manifest_file = definition_dir / DEFINITION_MANIFEST_FILE_NAME

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    developer_prompt = read_prompt(batch_name)

    definition_rows: list[dict] = []
    manifest_rows: list[dict] = []
    seen_ids: set[str] = set()
    skipped_rows: list[dict] = []

    for source_line_no, row in iter_jsonl(input_path):
        try:
            prompt_value = build_user_payload(dataset_folder, row, source_field_name)
        except ValueError as e:
            skipped_rows.append({"source_line_no": source_line_no, "reason": str(e)})
            continue

        unique_id = make_unique_id(batch_name, row, source_line_no, seen_ids)
        request_obj = make_request_object(unique_id, developer_prompt, prompt_value)

        definition_rows.append({"unique_id": unique_id, "request": request_obj})
        manifest_rows.append(
            {
                "unique_id": unique_id,
                "custom_id": unique_id,
                "source_line_no": source_line_no,
                "source_hash": row.get("source_hash"),
                "sample_index": row.get("sample_index"),
                "dataset_folder": dataset_folder,
                "source_field_name": source_field_name,
                "batch_name": batch_name,
                "original_obj": row,
            }
        )

    if not definition_rows:
        raise RuntimeError(f"No eligible rows found in {input_path}")

    write_jsonl(definition_file, definition_rows)
    write_jsonl(manifest_file, manifest_rows)

    print(f"Prepared batch definition: {definition_file}")
    print(f"Prepared batch manifest: {manifest_file}")
    print(f"Rows prepared: {len(definition_rows)}")
    if skipped_rows:
        print(f"Rows skipped: {len(skipped_rows)}")


def parse_response_object(response_text: str, unique_id: str) -> dict:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response_text for {unique_id} is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"response_text for {unique_id} must be a JSON object")
    return parsed


def collect_dataset(batch_name: str, merge_field_name: str | None = None) -> None:
    root = task_root()
    responses_path = resolve_batch_responses_path(batch_name)
    output_path = root / BATCHES_DIR_NAME / batch_name / DEFAULT_INPUT_FILE_NAME

    manifest_map = load_manifest_map(batch_name)
    response_rows = read_jsonl(responses_path)

    collected_rows: list[dict] = []
    missing_manifest = 0

    for response_row in response_rows:
        unique_id = str(response_row.get("unique_id") or response_row.get("custom_id") or "").strip()
        manifest_row = manifest_map.get(unique_id)
        if manifest_row is None:
            missing_manifest += 1
            continue

        original_obj = manifest_row.get("original_obj")
        if not isinstance(original_obj, dict):
            missing_manifest += 1
            continue

        merged = dict(original_obj)
        response_text = response_row.get("response_text")

        if merge_field_name:
            merged[merge_field_name] = response_text
        else:
            parsed = parse_response_object(str(response_text or ""), unique_id)
            merged.update(parsed)

        collected_rows.append(merged)

    if not collected_rows:
        raise RuntimeError("No rows were collected. Check batch responses and the manifest.")

    write_jsonl(output_path, collected_rows)
    print(f"Collected dataset: {output_path}")
    print(f"Rows collected: {len(collected_rows)}")
    if missing_manifest:
        print(f"Rows skipped (missing manifest/original_obj): {missing_manifest}")


def usage() -> str:
    return (
        "Usage:\n"
        "  python tasks.py batch <new_batch_name> <dataset_folder> [source_field_name]\n"
        "  python tasks.py collect <batch_name> [merge_field_name]\n\n"
        "Batch behavior:\n"
        "  - If source_field_name is provided, that field is sent directly as the user content.\n"
        "  - If source_field_name is omitted, <dataset_folder>/map is read and a JSON object is built from it.\n"
        "    Map format: ds_field=field_in_prompt\n\n"
        "Collect behavior:\n"
        "  - If merge_field_name is provided, raw response_text is stored under that field.\n"
        "  - If merge_field_name is omitted, response_text is parsed as a JSON object and its keys are merged.\n\n"
        "Examples:\n"
        "  python tasks.py batch translate ds text\n"
        "  python tasks.py batch score_translation translate\n"
        "  python tasks.py collect translate english_translation\n"
        "  python tasks.py collect score_translation\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise ValueError(usage())

    command = argv[1].strip()

    if command == "collect":
        if len(argv) not in {3, 4}:
            raise ValueError(usage())
        batch_name = argv[2].strip()
        merge_field_name = argv[3].strip() if len(argv) == 4 else None
        if not batch_name:
            raise ValueError(usage())
        collect_dataset(batch_name, merge_field_name)
        return 0

    if command == "batch":
        if len(argv) not in {4, 5}:
            raise ValueError(usage())
        batch_name = argv[2].strip()
        dataset_folder = argv[3].strip()
        source_field_name = argv[4].strip() if len(argv) == 5 else None
        if not batch_name or not dataset_folder:
            raise ValueError(usage())
        generate_batch(batch_name, dataset_folder, source_field_name)
        return 0

    raise ValueError(usage())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
