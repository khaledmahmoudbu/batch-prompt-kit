# BatchPromptKit

BatchPromptKit is a small utility for running repeatable batch LLM prompt workflows.

It helps you turn a JSONL dataset and a prompt into a batch job, run that job through the OpenAI Batch API, retry failures, and collect the model outputs back into a clean dataset.

It is useful for:

- testing fine-tuned models
- generating synthetic data
- scoring model outputs
- comparing prompts
- translating datasets
- simplifying datasets
- building multi-step LLM data pipelines

---

## Why this exists

LLM work often requires running the same prompt over many examples.

Doing this manually is slow and error-prone.

BatchPromptKit gives you a simple workflow:

```text
dataset + prompt
        ↓
batch definition
        ↓
OpenAI batch run
        ↓
merged responses
        ↓
collected dataset
``` 
---

## Minimal happy-case workflow

Assume this simple structure:

```text
batch-prompt-kit/
├── tasks.py
├── batch.py
├── key
├── prompts/
│   └── translate
└── ds/
    └── data.jsonl
```

Where:

- `key` contains your OpenAI API key
- `prompts/translate` contains the prompt
- `ds/data.jsonl` contains the dataset
- `translate` is the batch name
- `text` is the dataset field sent to the model

Install the OpenAI package:

```bash
python -m pip install -U openai
```

Create the batch definition:

```bash
python tasks.py batch translate ds text
```

This reads `prompts/translate` and `ds/data.jsonl`, then creates the batch request file.

Submit the batch:

```bash
python batch.py submit translate
```

This sends the generated batch to OpenAI.

Wait for the batch to finish:

```bash
python batch.py wait translate
```

This keeps checking the batch status until it completes.

Download the results:

```bash
python batch.py download translate
```

This downloads the completed output files.

Merge successful responses:

```bash
python batch.py merge translate
```

This creates the merged response file:

```text
batches/translate/batch_responses.jsonl
```

Collect responses into the final dataset:

```bash
python tasks.py collect translate translated_text
```

This stores each model response in the `translated_text` field.

Final output:

```text
batches/translate/data.jsonl
```

---

## Retry failed requests

If some requests fail, run:

```bash
python batch.py retry translate
```

Then repeat:

```bash
python batch.py wait translate
python batch.py download translate
python batch.py merge translate
python tasks.py collect translate translated_text
```

`retry` submits only failed requests from the previous downloaded run.


## Minimal happy-case workflow

Assume this simple structure:

```text
batch-prompt-kit/
├── tasks.py
├── batch.py
├── key
├── prompts/
│   └── translate
└── ds/
    └── data.jsonl
```

Where:

- `key` contains your OpenAI API key
- `prompts/translate` contains the prompt
- `ds/data.jsonl` contains the dataset
- `translate` is the batch name
- `text` is the dataset field sent to the model


Create the batch definition:

```bash
python tasks.py batch translate ds text
```

This reads `prompts/translate` and `ds/data.jsonl`, then creates the batch request file.

Submit the batch:

```bash
python batch.py submit translate
```

This sends the generated batch to OpenAI.

Wait for the batch to finish:

```bash
python batch.py wait translate
```

This keeps checking the batch status until it completes.

Download the results:

```bash
python batch.py download translate
```

This downloads the completed output files.

Merge successful responses:

```bash
python batch.py merge translate
```

This creates the merged response file:

```text
batches/translate/batch_responses.jsonl
```

Collect responses into the final dataset:

```bash
python tasks.py collect translate translated_text
```

This stores each model response in the `translated_text` field.

Final output:

```text
batches/translate/data.jsonl
```
