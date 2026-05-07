from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_POLL_SECONDS = 20
DEFINITION_DIR_NAME = "definition"
DEFINITION_FILE_NAME = "generated_batch.jsonl"
DEFINITION_MANIFEST_FILE_NAME = "generated_batch_manifest.jsonl"
RUNS_DIR_NAME = "runs"
DOWNLOADS_DIR_NAME = "downloads"
BATCHES_DIR_NAME = "batches"
MERGED_OUTPUT_FILE_NAME = "batch_responses.jsonl"
MERGE_SUMMARY_FILE_NAME = "merge_summary.json"
SUBMISSION_DIR_NAME = "submission"
SUBMISSION_REQUEST_FILE_NAME = "request_batch.jsonl"
SUBMISSION_MANIFEST_FILE_NAME = "submission_manifest.jsonl"
SUBMISSION_SUMMARY_FILE_NAME = "submission_summary.json"


@dataclass
class BatchPaths:
    task_dir: Path
    task_name: str
    batch_dir: Path
    batch_name: str
    definition_dir: Path
    definition_file: Path
    definition_manifest_file: Path
    runs_dir: Path


@dataclass
class DefinitionEntry:
    unique_id: str
    request_obj: dict
    source_definition_line_no: int


class RuntimeErrorWithHint(RuntimeError):
    pass


REQUEST_REQUIRED_KEYS = {"custom_id", "method", "url", "body"}
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def current_task_dir() -> Path:
    return Path.cwd().resolve()


def resolve_batch_paths(batch_name: str, task_dir: Optional[Path] = None) -> BatchPaths:
    task_dir = (task_dir or current_task_dir()).resolve()
    batch_dir = task_dir / BATCHES_DIR_NAME / batch_name
    definition_dir = batch_dir / DEFINITION_DIR_NAME
    definition_file = definition_dir / DEFINITION_FILE_NAME
    definition_manifest_file = definition_dir / DEFINITION_MANIFEST_FILE_NAME
    runs_dir = batch_dir / RUNS_DIR_NAME

    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch folder does not exist: {batch_dir}")
    if not definition_file.exists():
        raise FileNotFoundError(
            f"Missing definition file: {definition_file}. Expected to run this command from inside the task folder."
        )

    runs_dir.mkdir(parents=True, exist_ok=True)

    return BatchPaths(
        task_dir=task_dir,
        task_name=task_dir.name,
        batch_dir=batch_dir,
        batch_name=batch_name,
        definition_dir=definition_dir,
        definition_file=definition_file,
        definition_manifest_file=definition_manifest_file,
        runs_dir=runs_dir,
    )


def list_batch_names(task_dir: Optional[Path] = None) -> List[str]:
    task_dir = (task_dir or current_task_dir()).resolve()
    batch_names: List[str] = []
    batches_root = task_dir / BATCHES_DIR_NAME
    if not batches_root.exists():
        return []
    for p in sorted([x for x in batches_root.iterdir() if x.is_dir()]):
        if (p / DEFINITION_DIR_NAME / DEFINITION_FILE_NAME).exists():
            batch_names.append(p.name)
    return batch_names


def ensure_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeErrorWithHint(
            "Missing dependency 'openai'. Install it with: python -m pip install -U openai"
        ) from exc

    key_file = current_task_dir() / "key"
    if not key_file.exists():
        raise RuntimeErrorWithHint(f"Missing key file: {key_file}")

    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeErrorWithHint(f"Key file is empty: {key_file}")

    return OpenAI(api_key=api_key)


def api_call_with_retries(fn, *args, max_retries: int = 5, **kwargs):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_s = min(2 ** attempt, 20)
            print(
                f"API call failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {sleep_s}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise last_exc


def _file_content_to_bytes(content_obj) -> bytes:
    if isinstance(content_obj, bytes):
        return content_obj
    if isinstance(content_obj, str):
        return content_obj.encode("utf-8")
    if hasattr(content_obj, "content"):
        raw = content_obj.content
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return raw.encode("utf-8")
    if hasattr(content_obj, "read"):
        raw = content_obj.read()
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return raw.encode("utf-8")
    return str(content_obj).encode("utf-8")


def normalize_definition_row(row: dict, line_no: int) -> DefinitionEntry:
    if "request" in row:
        unique_id = str(row.get("unique_id") or "").strip()
        request_obj = row["request"]
    else:
        request_obj = row
        unique_id = str(row.get("unique_id") or row.get("custom_id") or "").strip()

    if not unique_id:
        raise ValueError(f"Definition line {line_no}: missing unique_id")
    if not isinstance(request_obj, dict):
        raise ValueError(f"Definition line {line_no}: request must be a JSON object")

    request_obj = dict(request_obj)
    custom_id = str(request_obj.get("custom_id") or "").strip()
    if not custom_id:
        request_obj["custom_id"] = unique_id
        custom_id = unique_id

    missing = REQUEST_REQUIRED_KEYS - set(request_obj.keys())
    if missing:
        raise ValueError(
            f"Definition line {line_no}: request object is missing required keys: {sorted(missing)}"
        )

    if custom_id != unique_id:
        raise ValueError(
            f"Definition line {line_no}: custom_id ({custom_id}) must match unique_id ({unique_id})"
        )

    return DefinitionEntry(
        unique_id=unique_id,
        request_obj=request_obj,
        source_definition_line_no=line_no,
    )


def load_definition_entries(definition_file: Path) -> List[DefinitionEntry]:
    entries: List[DefinitionEntry] = []
    seen: set[str] = set()
    for line_no, row in iter_jsonl(definition_file):
        entry = normalize_definition_row(row, line_no)
        if entry.unique_id in seen:
            raise ValueError(f"Duplicate unique_id/custom_id in definition file: {entry.unique_id}")
        seen.add(entry.unique_id)
        entries.append(entry)
    if not entries:
        raise ValueError(f"Definition file is empty: {definition_file}")
    return entries


def run_dir_name(run_id: int) -> str:
    return f"run_{run_id:04d}"


def run_id_from_dir_name(name: str) -> Optional[int]:
    m = re.fullmatch(r"run_(\d{4})", name)
    return int(m.group(1)) if m else None


def next_run_id(runs_dir: Path) -> int:
    existing = [
        run_id_from_dir_name(p.name)
        for p in runs_dir.iterdir()
        if p.is_dir() and run_id_from_dir_name(p.name) is not None
    ]
    return (max(existing) if existing else 0) + 1


def resolve_run_dir(paths: BatchPaths, selector: Optional[str]) -> Path:
    if selector and selector != "all":
        try:
            run_id = int(selector)
        except ValueError as exc:
            raise RuntimeErrorWithHint(f"Invalid run selector: {selector}") from exc
        run_dir = paths.runs_dir / run_dir_name(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run does not exist: {run_dir}")
        return run_dir

    candidates = [
        p for p in paths.runs_dir.iterdir() if p.is_dir() and run_id_from_dir_name(p.name) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No runs found under: {paths.runs_dir}")
    return sorted(candidates, key=lambda p: run_id_from_dir_name(p.name) or 0)[-1]


def all_run_dirs(paths: BatchPaths) -> List[Path]:
    return [
        p
        for p in sorted(paths.runs_dir.iterdir(), key=lambda p: run_id_from_dir_name(p.name) or 0)
        if p.is_dir() and run_id_from_dir_name(p.name) is not None
    ]


def make_run_skeleton(paths: BatchPaths, run_id: int) -> Path:
    run_dir = paths.runs_dir / run_dir_name(run_id)
    (run_dir / SUBMISSION_DIR_NAME).mkdir(parents=True, exist_ok=False)
    (run_dir / DOWNLOADS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_submission_manifest(
    entries: List[DefinitionEntry],
    *,
    task_name: str,
    batch_name: str,
    run_id: int,
    retry_of_run_id: Optional[int],
) -> List[dict]:
    rows: List[dict] = []
    for entry in entries:
        rows.append(
            {
                "task_name": task_name,
                "batch_name": batch_name,
                "run_id": run_id,
                "retry_of_run_id": retry_of_run_id,
                "attempt_no": run_id,
                "unique_id": entry.unique_id,
                "custom_id": entry.request_obj["custom_id"],
                "source_definition_line_no": entry.source_definition_line_no,
            }
        )
    return rows


def create_local_run_from_entries(
    paths: BatchPaths,
    entries: List[DefinitionEntry],
    *,
    retry_of_run_id: Optional[int] = None,
) -> Tuple[int, Path, dict]:
    run_id = next_run_id(paths.runs_dir)
    run_dir = make_run_skeleton(paths, run_id)

    submission_dir = run_dir / SUBMISSION_DIR_NAME
    request_file = submission_dir / SUBMISSION_REQUEST_FILE_NAME
    request_manifest_file = submission_dir / SUBMISSION_MANIFEST_FILE_NAME
    summary_file = submission_dir / SUBMISSION_SUMMARY_FILE_NAME

    write_jsonl(request_file, [entry.request_obj for entry in entries])
    write_jsonl(
        request_manifest_file,
        build_submission_manifest(
            entries,
            task_name=paths.task_name,
            batch_name=paths.batch_name,
            run_id=run_id,
            retry_of_run_id=retry_of_run_id,
        ),
    )

    summary = {
        "task_name": paths.task_name,
        "batch_name": paths.batch_name,
        "run_id": run_id,
        "attempt_no": run_id,
        "retry_of_run_id": retry_of_run_id,
        "created_at_utc": utc_now_iso(),
        "launch_datetime_utc": None,
        "source_definition_file": str(paths.definition_file.relative_to(paths.task_dir)),
        "source_definition_manifest_file": (
            str(paths.definition_manifest_file.relative_to(paths.task_dir))
            if paths.definition_manifest_file.exists()
            else None
        ),
        "request_count": len(entries),
        "submission_request_file": str(request_file.relative_to(paths.task_dir)),
        "submission_manifest_file": str(request_manifest_file.relative_to(paths.task_dir)),
        "downloads_dir": str((run_dir / DOWNLOADS_DIR_NAME).relative_to(paths.task_dir)),
        "openai": {
            "input_file_id": None,
            "batch_id": None,
            "status": "local_prepared",
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": {"total": None, "completed": None, "failed": None},
        },
    }
    write_json(summary_file, summary)
    return run_id, run_dir, summary


def load_run_summary(run_dir: Path) -> dict:
    summary_file = run_dir / SUBMISSION_DIR_NAME / SUBMISSION_SUMMARY_FILE_NAME
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing submission summary: {summary_file}")
    return read_json(summary_file)


def save_run_summary(run_dir: Path, summary: dict) -> None:
    write_json(run_dir / SUBMISSION_DIR_NAME / SUBMISSION_SUMMARY_FILE_NAME, summary)


def print_run_summary(summary: dict) -> None:
    openai = summary.get("openai") or {}
    counts = openai.get("request_counts") or {}
    print(
        f"batch={summary.get('batch_name')} | run_id={summary.get('run_id')} | status={openai.get('status')} | "
        f"batch_id={openai.get('batch_id')} | completed={counts.get('completed')} | "
        f"failed={counts.get('failed')} | total={counts.get('total')}"
    )


def submit_batch(batch_name: str, task_dir: Optional[Path] = None) -> dict:
    paths = resolve_batch_paths(batch_name, task_dir)
    entries = load_definition_entries(paths.definition_file)
    run_id, run_dir, summary = create_local_run_from_entries(paths, entries, retry_of_run_id=None)

    client = ensure_openai_client()
    request_file = run_dir / SUBMISSION_DIR_NAME / SUBMISSION_REQUEST_FILE_NAME
    with request_file.open("rb") as f:
        input_file = api_call_with_retries(client.files.create, file=f, purpose="batch")

    batch = api_call_with_retries(
        client.batches.create,
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=DEFAULT_COMPLETION_WINDOW,
        metadata={
            "task_name": paths.task_name,
            "batch_name": paths.batch_name,
            "run_id": str(run_id),
            "request_file": request_file.name,
        },
    )

    summary["launch_datetime_utc"] = utc_now_iso()
    summary["openai"] = {
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "total": getattr(getattr(batch, "request_counts", None), "total", None),
            "completed": getattr(getattr(batch, "request_counts", None), "completed", None),
            "failed": getattr(getattr(batch, "request_counts", None), "failed", None),
        },
    }
    save_run_summary(run_dir, summary)
    print(f"Submitted {paths.batch_name} -> {run_dir_name(run_id)}")
    print_run_summary(summary)
    return summary


def refresh_one_run_status(run_dir: Path) -> dict:
    summary = load_run_summary(run_dir)
    batch_id = ((summary.get("openai") or {}).get("batch_id"))
    if not batch_id:
        print_run_summary(summary)
        return summary

    client = ensure_openai_client()
    batch = api_call_with_retries(client.batches.retrieve, batch_id)
    counts = getattr(batch, "request_counts", None)
    summary["openai"]["status"] = batch.status
    summary["openai"]["output_file_id"] = getattr(batch, "output_file_id", None)
    summary["openai"]["error_file_id"] = getattr(batch, "error_file_id", None)
    summary["openai"]["request_counts"] = {
        "total": getattr(counts, "total", None),
        "completed": getattr(counts, "completed", None),
        "failed": getattr(counts, "failed", None),
    }
    summary["last_status_check_utc"] = utc_now_iso()
    save_run_summary(run_dir, summary)
    print_run_summary(summary)
    return summary


def status_all_batches(task_dir: Optional[Path] = None) -> None:
    names = list_batch_names(task_dir)
    if not names:
        print("No batch folders found in the current task folder.")
        return
    for batch_name in names:
        try:
            paths = resolve_batch_paths(batch_name, task_dir)
            run_dir = resolve_run_dir(paths, None)
            refresh_one_run_status(run_dir)
        except Exception as exc:
            print(f"batch={batch_name} | error={exc}")


def status_batch(batch_name: Optional[str], selector: Optional[str], task_dir: Optional[Path] = None) -> None:
    if not batch_name:
        status_all_batches(task_dir)
        return

    paths = resolve_batch_paths(batch_name, task_dir)
    if selector == "all":
        run_dirs = all_run_dirs(paths)
        if not run_dirs:
            raise FileNotFoundError(f"No runs found under: {paths.runs_dir}")
        for run_dir in run_dirs:
            refresh_one_run_status(run_dir)
        return

    run_dir = resolve_run_dir(paths, selector)
    refresh_one_run_status(run_dir)


def wait_batch(batch_name: str, selector: Optional[str], poll_seconds: int = DEFAULT_POLL_SECONDS, task_dir: Optional[Path] = None) -> None:
    paths = resolve_batch_paths(batch_name, task_dir)
    run_dir = resolve_run_dir(paths, selector)
    while True:
        summary = refresh_one_run_status(run_dir)
        status = ((summary.get("openai") or {}).get("status"))
        if status in TERMINAL_STATUSES or status in {"local_prepared"}:
            print("Run reached a terminal/local state.")
            return
        print(f"Sleeping {poll_seconds}s...")
        time.sleep(poll_seconds)


def download_one_run(run_dir: Path) -> dict:
    summary = refresh_one_run_status(run_dir)
    openai = summary.get("openai") or {}
    batch_id = openai.get("batch_id")
    if not batch_id:
        print(f"Skipping download for {run_dir.name}: no OpenAI batch_id present.")
        return summary

    client = ensure_openai_client()
    downloads_dir = run_dir / DOWNLOADS_DIR_NAME
    downloads_dir.mkdir(parents=True, exist_ok=True)

    output_file_id = openai.get("output_file_id")
    if output_file_id:
        out_path = downloads_dir / "output.jsonl"
        if not out_path.exists():
            raw = api_call_with_retries(client.files.content, output_file_id)
            out_path.write_bytes(_file_content_to_bytes(raw))
            print(f"Downloaded output -> {out_path}")

    error_file_id = openai.get("error_file_id")
    if error_file_id:
        err_path = downloads_dir / "errors.jsonl"
        if not err_path.exists():
            raw = api_call_with_retries(client.files.content, error_file_id)
            err_path.write_bytes(_file_content_to_bytes(raw))
            print(f"Downloaded errors -> {err_path}")

    return summary


def download_batch(batch_name: str, selector: Optional[str], task_dir: Optional[Path] = None) -> None:
    paths = resolve_batch_paths(batch_name, task_dir)
    if selector == "all":
        run_dirs = all_run_dirs(paths)
        if not run_dirs:
            raise FileNotFoundError(f"No runs found under: {paths.runs_dir}")
        for run_dir in run_dirs:
            download_one_run(run_dir)
        return

    run_dir = resolve_run_dir(paths, selector)
    download_one_run(run_dir)


def is_batch_row_transport_success(row: dict) -> bool:
    if row.get("error"):
        return False
    response = row.get("response") or {}
    status_code = response.get("status_code")
    if not isinstance(status_code, int) or not (200 <= status_code < 300):
        return False
    body = response.get("body") or {}
    if body.get("error"):
        return False
    choices = body.get("choices") or []
    if not choices:
        return False
    finish_reason = choices[0].get("finish_reason")
    if finish_reason in {"length", "content_filter"}:
        return False
    return True


def extract_assistant_text_from_batch_row(row: dict) -> str:
    response = row.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("No choices in response body")
    message = choices[0].get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        raise ValueError(f"Model refusal: {refusal}")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content or "")


def load_submission_entries_for_run(run_dir: Path) -> List[DefinitionEntry]:
    request_file = run_dir / SUBMISSION_DIR_NAME / SUBMISSION_REQUEST_FILE_NAME
    manifest_file = run_dir / SUBMISSION_DIR_NAME / SUBMISSION_MANIFEST_FILE_NAME
    manifest_by_custom_id: Dict[str, dict] = {}
    if manifest_file.exists():
        for _, row in iter_jsonl(manifest_file):
            manifest_by_custom_id[str(row.get("custom_id"))] = row

    entries: List[DefinitionEntry] = []
    for line_no, row in iter_jsonl(request_file):
        custom_id = str(row.get("custom_id") or "").strip()
        manifest_row = manifest_by_custom_id.get(custom_id, {})
        unique_id = str(manifest_row.get("unique_id") or custom_id).strip()
        source_definition_line_no = int(manifest_row.get("source_definition_line_no") or line_no)
        entries.append(
            DefinitionEntry(
                unique_id=unique_id,
                request_obj=row,
                source_definition_line_no=source_definition_line_no,
            )
        )
    return entries


def compute_retry_entries_from_run(
    run_dir: Path,
    *,
    response_validator: Optional[Callable[[str], None]] = None,
) -> List[DefinitionEntry]:
    downloads_dir = run_dir / DOWNLOADS_DIR_NAME
    output_file = downloads_dir / "output.jsonl"
    error_file = downloads_dir / "errors.jsonl"

    if not output_file.exists() and not error_file.exists():
        raise FileNotFoundError(
            f"No downloaded artifacts found under {downloads_dir}. Run download before retry."
        )

    submitted_entries = load_submission_entries_for_run(run_dir)
    submitted_by_custom_id = {entry.request_obj["custom_id"]: entry for entry in submitted_entries}

    successful: set[str] = set()
    failed: set[str] = set()

    if output_file.exists():
        for _, row in iter_jsonl(output_file):
            custom_id = str(row.get("custom_id") or "").strip()
            if not custom_id:
                continue
            if not is_batch_row_transport_success(row):
                failed.add(custom_id)
                continue
            if response_validator is not None:
                try:
                    text = extract_assistant_text_from_batch_row(row)
                    response_validator(text)
                except Exception:
                    failed.add(custom_id)
                    continue
            successful.add(custom_id)

    if error_file.exists():
        for _, row in iter_jsonl(error_file):
            custom_id = str(row.get("custom_id") or "").strip()
            if custom_id:
                failed.add(custom_id)

    retry_custom_ids = sorted(cid for cid in failed if cid and cid not in successful)
    return [submitted_by_custom_id[cid] for cid in retry_custom_ids if cid in submitted_by_custom_id]


def retry_batch(batch_name: str, selector: Optional[str], task_dir: Optional[Path] = None) -> None:
    paths = resolve_batch_paths(batch_name, task_dir)
    source_run_dir = resolve_run_dir(paths, selector)
    source_summary = load_run_summary(source_run_dir)
    retry_entries = compute_retry_entries_from_run(source_run_dir, response_validator=None)
    if not retry_entries:
        print(f"No retry candidates found for {batch_name} from {source_run_dir.name}.")
        return

    new_run_id, new_run_dir, summary = create_local_run_from_entries(
        paths,
        retry_entries,
        retry_of_run_id=int(source_summary.get("run_id")),
    )

    client = ensure_openai_client()
    request_file = new_run_dir / SUBMISSION_DIR_NAME / SUBMISSION_REQUEST_FILE_NAME
    with request_file.open("rb") as f:
        input_file = api_call_with_retries(client.files.create, file=f, purpose="batch")

    batch = api_call_with_retries(
        client.batches.create,
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=DEFAULT_COMPLETION_WINDOW,
        metadata={
            "task_name": paths.task_name,
            "batch_name": paths.batch_name,
            "run_id": str(new_run_id),
            "retry_of_run_id": str(source_summary.get("run_id")),
            "request_file": request_file.name,
        },
    )

    summary["launch_datetime_utc"] = utc_now_iso()
    summary["openai"] = {
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "total": getattr(getattr(batch, "request_counts", None), "total", None),
            "completed": getattr(getattr(batch, "request_counts", None), "completed", None),
            "failed": getattr(getattr(batch, "request_counts", None), "failed", None),
        },
    }
    save_run_summary(new_run_dir, summary)
    print(f"Submitted retry for {batch_name} from {source_run_dir.name} -> {run_dir_name(new_run_id)}")
    print_run_summary(summary)


def collect_successes_from_run(run_dir: Path) -> List[dict]:
    summary = load_run_summary(run_dir)
    output_file = run_dir / DOWNLOADS_DIR_NAME / "output.jsonl"
    if not output_file.exists():
        return []

    submission_entries = load_submission_entries_for_run(run_dir)
    by_custom_id = {entry.request_obj["custom_id"]: entry for entry in submission_entries}

    successes: List[dict] = []
    for _, row in iter_jsonl(output_file):
        custom_id = str(row.get("custom_id") or "").strip()
        if not custom_id or not is_batch_row_transport_success(row):
            continue

        entry = by_custom_id.get(custom_id)
        unique_id = entry.unique_id if entry else custom_id
        source_definition_line_no = entry.source_definition_line_no if entry else None

        try:
            response_text = extract_assistant_text_from_batch_row(row)
        except Exception:
            response_text = None

        successes.append(
            {
                "task_name": summary.get("task_name"),
                "batch_name": summary.get("batch_name"),
                "run_id": summary.get("run_id"),
                "attempt_no": summary.get("attempt_no"),
                "retry_of_run_id": summary.get("retry_of_run_id"),
                "unique_id": unique_id,
                "custom_id": custom_id,
                "source_definition_line_no": source_definition_line_no,
                "response_text": response_text,
                "raw_result": row,
            }
        )
    return successes


def merge_batch(batch_name: str, task_dir: Optional[Path] = None) -> None:
    paths = resolve_batch_paths(batch_name, task_dir)
    run_dirs = all_run_dirs(paths)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under: {paths.runs_dir}")

    latest_success_by_unique_id: Dict[str, dict] = {}
    scanned_runs: List[int] = []
    runs_missing_output: List[int] = []

    for run_dir in run_dirs:
        run_id = run_id_from_dir_name(run_dir.name)
        if run_id is None:
            continue
        scanned_runs.append(run_id)
        output_file = run_dir / DOWNLOADS_DIR_NAME / "output.jsonl"
        if not output_file.exists():
            runs_missing_output.append(run_id)
            continue
        for row in collect_successes_from_run(run_dir):
            unique_id = str(row.get("unique_id") or "").strip()
            if unique_id:
                latest_success_by_unique_id[unique_id] = row

    merged_rows = sorted(
        latest_success_by_unique_id.values(),
        key=lambda r: (
            10**18 if r.get("source_definition_line_no") is None else int(r.get("source_definition_line_no")),
            str(r.get("unique_id") or ""),
        ),
    )

    merged_output_file = paths.batch_dir / MERGED_OUTPUT_FILE_NAME
    merge_summary_file = paths.batch_dir / MERGE_SUMMARY_FILE_NAME
    write_jsonl(merged_output_file, merged_rows)
    write_json(
        merge_summary_file,
        {
            "task_name": paths.task_name,
            "batch_name": paths.batch_name,
            "merged_at_utc": utc_now_iso(),
            "scanned_run_ids": scanned_runs,
            "runs_missing_output": runs_missing_output,
            "merged_row_count": len(merged_rows),
            "merged_output_file": str(merged_output_file.relative_to(paths.task_dir)),
        },
    )
    print(f"Merged successful outcomes for {batch_name} -> {merged_output_file} | rows={len(merged_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run this from inside a task folder. Example: cd tasks/evaluate && python ../../batch_runtime_simple.py submit meaning"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("batch")

    status = sub.add_parser("status")
    status.add_argument("batch", nargs="?")
    status.add_argument("selector", nargs="?")

    wait = sub.add_parser("wait")
    wait.add_argument("batch")
    wait.add_argument("selector", nargs="?")
    wait.add_argument("poll_seconds", nargs="?", type=int, default=DEFAULT_POLL_SECONDS)

    download = sub.add_parser("download")
    download.add_argument("batch")
    download.add_argument("selector", nargs="?")

    retry = sub.add_parser("retry")
    retry.add_argument("batch")
    retry.add_argument("selector", nargs="?")

    merge = sub.add_parser("merge")
    merge.add_argument("batch")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "submit":
            submit_batch(args.batch)
        elif args.command == "status":
            status_batch(args.batch, args.selector)
        elif args.command == "wait":
            wait_batch(args.batch, args.selector, args.poll_seconds)
        elif args.command == "download":
            download_batch(args.batch, args.selector)
        elif args.command == "retry":
            retry_batch(args.batch, args.selector)
        elif args.command == "merge":
            merge_batch(args.batch)
        else:
            parser.print_help()
            return 1
    except RuntimeErrorWithHint as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
