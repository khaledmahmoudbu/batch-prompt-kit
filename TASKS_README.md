## What `tasks.py` does

`tasks.py` is the workflow layer for BatchPromptKit.

It prepares dataset rows to be processed by an LLM batch job, then collects the completed responses back into a usable JSONL dataset.

At a high level, it supports two main actions:

1. **Create a batch definition**
   - Reads examples from a dataset folder.
   - Reads the matching prompt from the `prompts/` directory.
   - Builds one LLM request per dataset row.
   - Assigns each request a unique `custom_id`.
   - Generates a batch request file.
   - Generates a manifest that remembers where every request came from.

2. **Collect batch results**
   - Reads completed batch responses.
   - Uses the manifest to match each response back to the original dataset row.
   - Either stores the raw model response in a new field or parses the response as JSON and merges it into the dataset.
   - Writes the collected dataset as a new `data.jsonl` file inside the batch folder.

`tasks.py` does not run the batch job itself. Its job is to prepare clean batch inputs and reconstruct clean datasets after the batch responses are available.

## Directory structure

Before generating a batch, you only need to create the dataset folder and the matching prompt file.

```text
batch-prompt-kit/
├── tasks.py
├── batch.py
├── prompts/
│   └── <batch_name>
└── <dataset_folder>/
    ├── data.jsonl
    └── map ##  # optional, only needed for mapped multi-field prompts
```

After you generate a batch, BatchPromptKit automatically creates a new folder under `batches/` using the same batch name.

```text
batches/
└── <batch_name>/
    └── definition/
        ├── generated_batch.jsonl
        └── generated_batch_manifest.jsonl
```

## How batch generation works

To generate a batch, run:

```bash
python tasks.py batch <batch_name> <dataset_folder> [source_field_name]
```

`tasks.py` reads the prompt from:

```text
prompts/<batch_name>
```

It also reads the dataset from:

```text
<dataset_folder>/data.jsonl
```

Then it creates one batch request for each valid dataset row.

Each request contains:

- the prompt as the developer message
- the prompt parameters/field filled from the corresponding dataset row
- a unique `custom_id`
- the model settings used for the request

## Single-field prompts

If your prompt only needs one input field, pass that field name directly.

Example:

```bash
python tasks.py batch translate ds text
```

Example dataset row:

```json
{"text": "The city opened a new clinic."}
```

In this case, the value of `text` is sent directly to the model as the user message.

## Multi-field prompts and the `map` file

If your prompt needs multiple input fields, do not pass `source_field_name`.

Example:

```bash
python tasks.py batch score_translation translations
```

In this case, `tasks.py` reads:

```text
translations/map
```

The `map` file tells `tasks.py` which dataset fields should be sent to the prompt.

Map format:

```text
dataset_field=prompt_field
```

Example `map` file:

```text
original_text=source
translated_text=translation
```

Example dataset row:

```json
{"original_text": "The city opened a new clinic.", "translated_text": "La ciudad abrió una nueva clínica."}
```

The model receives a JSON user message like:

```json
{
  "source": "The city opened a new clinic.",
  "translation": "La ciudad abrió una nueva clínica."
}
```

This is useful when the prompt needs more than one value, such as scoring, comparison, evaluation, or multi-input transformations.

## Collect batch results

After the batch job finishes, place the completed response file here:

```text
batches/<batch_name>/batch_responses.jsonl
```

Then run:

```bash
python tasks.py collect <batch_name> <merge_field_name>
```

Example:

```bash
python tasks.py collect translate simplified_text
```

This stores each model response under the field `simplified_text`.

The final collected dataset is saved here:

```text
batches/<batch_name>/data.jsonl
```

Example output row:

```json
{"text": "The city opened a new clinic.", "simplified_text": "The city opened a new clinic."}
```