# `batch.py`

`batch.py` is the runtime manager for OpenAI batch jobs.

It submits generated batch request files, tracks their status, downloads results, retries failed requests, and merges successful responses across runs.

---

## What `batch.py` does

`batch.py` helps you:

- submit a batch to the OpenAI Batch API
- create numbered run folders like `run_0001`, `run_0002`, etc.
- check the status of submitted runs
- wait until a run finishes
- download output and error files
- retry failed requests in a new run
- merge successful responses across runs

---

## Why multiple runs exist

A batch can have many requests.

Sometimes, some requests may fail while others succeed.

Instead of submitting the entire batch again, `batch.py` supports retries.

A retry creates a new run with only the failed requests from a previous run.

Example:

```text
run_0001  -> original OpenAI batch
run_0002  -> retry OpenAI batch for failed requests
run_0003  -> another retry, if needed
```

Each run is a separate OpenAI batch job.

This makes retries cleaner and avoids repeating successful work.

---

## Required files

Before using `batch.py`, the batch definition must already exist:

```text
batches/<batch_name>/definition/generated_batch.jsonl
```

The definition file contains the requests that will be submitted.

You also need a `key` file in the current folder:

```text
key
```

The `key` file should contain your OpenAI API key.

You also need the OpenAI Python package:

```bash
python -m pip install -U openai
```

---

## Submit a batch

To submit a batch:

```bash
python batch.py submit <batch_name>
```

Example:

```bash
python batch.py submit translate
```

This creates a new run folder:

```text
batches/translate/runs/run_0001/
```

The run folder contains:

```text
submission/request_batch.jsonl
submission/submission_manifest.jsonl
submission/submission_summary.json
downloads/
```

Important:

```text
submit = submits the full batch definition
```

If you run `submit` again, it creates a new run and submits the full batch again.

It does not submit only failed rows.

---

## Check status

To check the latest run for one batch:

```bash
python batch.py status <batch_name>
```

Example:

```bash
python batch.py status translate
```

To check a specific run:

```bash
python batch.py status translate 1
```

To check all runs for a batch:

```bash
python batch.py status translate all
```

To check the latest run for every batch:

```bash
python batch.py status
```

---

## Wait for completion

To wait for the latest run:

```bash
python batch.py wait <batch_name>
```

Example:

```bash
python batch.py wait translate
```

To wait for a specific run:

```bash
python batch.py wait translate 1
```

By default, it checks every 20 seconds.

You can set a custom polling interval:

```bash
python batch.py wait translate 1 60
```

This checks run `1` every 60 seconds.

---

## Download results

After a batch finishes, download the results:

```bash
python batch.py download <batch_name>
```

Example:

```bash
python batch.py download translate
```

This downloads files into:

```text
batches/translate/runs/run_0001/downloads/
```

Possible files:

```text
output.jsonl
errors.jsonl
```

`output.jsonl` contains successful OpenAI batch responses.

`errors.jsonl` contains failed requests, if any exist.

To download a specific run:

```bash
python batch.py download translate 1
```

To download all runs:

```bash
python batch.py download translate all
```

---

## Retry failed requests

To retry failed requests from the latest run:

```bash
python batch.py retry <batch_name>
```

Example:

```bash
python batch.py retry translate
```

This reads the downloaded files from the selected run:

```text
downloads/output.jsonl
downloads/errors.jsonl
```

Then it creates a new run containing only failed requests.

Important:

```text
retry = submits failed requests from a previous downloaded run
```

You should run `download` before running `retry`.

---

## Merge successful responses

After downloading one or more runs, merge the successful responses:

```bash
python batch.py merge <batch_name>
```

Example:

```bash
python batch.py merge translate
```

This scans all run folders:

```text
batches/translate/runs/
```

Then it writes:

```text
batches/translate/batch_responses.jsonl
batches/translate/merge_summary.json
```

`batch_responses.jsonl` contains the latest successful response for each request.

If a request succeeded in a retry run, that newer success replaces the earlier failed attempt.

---

## Typical workflow

```bash
python batch.py submit translate
python batch.py status translate
python batch.py wait translate
python batch.py download translate
python batch.py merge translate
```

If some requests fail:

```bash
python batch.py retry translate
python batch.py wait translate
python batch.py download translate
python batch.py merge translate
```

---

## Key behavior
```text
submit   = submit the full batch definition
status   = check run status
wait     = keep checking until the run finishes
download = download output and error files
retry    = submit failed requests from a downloaded run
merge    = combine successful outputs across runs
```
