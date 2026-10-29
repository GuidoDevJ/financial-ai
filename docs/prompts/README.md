# Prompts (Amazon Bedrock Prompt Management)

The application does **not** read prompt files at runtime. Both system prompts live in Amazon
Bedrock Prompt Management and are fetched with `GetPrompt` (cached in memory, see below).

`registry.yaml` in this folder is the **hand-maintained source of truth and changelog**, same idea
as `ia-comafi-infra/service/src/prompts/registry.yaml`. Each prompt maps 1:1 to a Bedrock
`CreatePrompt` request: `name`, `description`, `defaultVariant` and a `TEXT` variant whose body is
`templateConfiguration.text.text`, with `inputVariables` listing every `{{variable}}` of the body.
Keep it identical to what is published in Bedrock.

The prompts are written as Bedrock **text prompts**: everything dynamic is a `{{variable}}` (the
console lets you fill them in its *Test variables* section), and the app fills the same variables
at call time:

| Prompt | Variables | Filled by |
| --- | --- | --- |
| `chat` | `banco`, `fecha_actual` | `BANK_NAME` setting; today's date (`YYYY-MM-DD`, Buenos Aires) computed on every turn |
| `brief` | `banco`, `fecha`, `cartera`, `precios`, `macro`, `noticias` | `BANK_NAME`; the brief date; the resolved data blocks as compact JSON (`"no disponible"` markers kept) |

Rendering is strict and single pass (`PromptVersion.render`): a `{{variable}}` the app does not
provide is a `PROMPT_VARIABLES_MISSING` error (502), extra values are ignored, and values are
inserted verbatim, so a news text containing `{{banco}}` is NOT substituted. The brief's user turn
is the fixed trigger `Generá el brief.` (plus the violation list on the retry). So do not add
variables to the published prompt unless the code provides them.

| Prompt | Used by | Env var |
| --- | --- | --- |
| `chat` | `POST /chat` (system prompt of every turn) | `BEDROCK_PROMPT_CHAT_ID` |
| `brief` | brief generator (CLI, scheduler, `/brief/preview`) | `BEDROCK_PROMPT_BRIEF_ID` |

One global `BEDROCK_PROMPT_VERSION` (default `DRAFT`) selects the version read for every prompt.
`registry.yaml` does not declare `modelId` or `inferenceConfiguration`: the app ignores them (it
reads only the text and calls the model with its own `BEDROCK_*` settings) and they only add risk
of `CreatePrompt` rejection.

Creating AWS resources is a **manual step for you**: nothing in this repository calls
`CreatePrompt` or `CreatePromptVersion`.

## Create the prompts (two ways)

Do this once per prompt, in the same region as `AWS_REGION`.

### A. Console

1. Amazon Bedrock console > **Prompt management** > **Create prompt**.
2. Name it as in `registry.yaml` (`financial-ai-chat`, `financial-ai-brief`).
3. Prompt type: **Text** (not Chat). One variant (`default`).
4. Paste the `text` of the prompt's `current_version` from `registry.yaml`. The console detects the
   `{{variables}}` itself (they must match `inputVariables` in the file; no model needed).
5. **Test it**: fill the *Test variables* with the values of
   [`test-variables.md`](test-variables.md) and run it (see that file for what to check).
6. Save, then **Create version**. AWS assigns the number (1, 2, ...); it may differ from the
   registry's own numbering, and that is the number for `BEDROCK_PROMPT_VERSION`.
7. Copy the prompt **ID** (10 characters, e.g. `ABCDEFGHIJ`) or its ARN.

### B. Exported CLI payload (offline generator)

```powershell
uv run python -m app.prompts.export --name chat > chat.json    # JSON on stdout, notes on stderr
aws bedrock-agent create-prompt --cli-input-json file://chat.json
aws bedrock-agent create-prompt-version --prompt-identifier <id from the previous output>
```

The export is generated from `registry.yaml` (`current_version`), makes no AWS call, and its shape
is validated against botocore's `CreatePrompt` model in the tests. Repeat with `--name brief`.

### Then set the env vars (`.env`)

```ini
BEDROCK_PROMPT_CHAT_ID=<chat prompt id or arn>
BEDROCK_PROMPT_BRIEF_ID=<brief prompt id or arn>
BEDROCK_PROMPT_VERSION=<number AWS assigned on "Create version">
PROMPT_CACHE_TTL_SECONDS=300
BANK_NAME=<bank name for {{banco}}>   # default "Banco Comafi" (an assumption: set it)
```

Verify without starting the API: `uv run python -m app.prompts.check`. It prints name, id,
version, hash, length and `drift=MATCH|DRIFT|NO_REFERENCE`, comparing the published text with
`registry.yaml`. Drift only warns (exit 0); exit is 1 only when a prompt cannot be fetched.

## Versioning workflow

1. Add a new entry under `versions` in `registry.yaml` (next number, with `changelog` and
   `variants`); keep the old ones as history. Set `current_version` to it. If you add or rename a
   `{{variable}}`, the code must provide it (`app/agents/chat_graph.py`, `app/brief/generate.py`)
   and `inputVariables` and `test-variables.md` must list it (the tests check this).
2. Publish it in Bedrock (edit the draft in the console, or `update-prompt`), test it with the
   *Test variables*, and **Create version**; AWS assigns the number.
3. Set `BEDROCK_PROMPT_VERSION` to that number and restart (or wait for
   `PROMPT_CACHE_TTL_SECONDS`).
4. Run `uv run python -m app.prompts.check` and confirm `drift=MATCH`.

Rollback: set `BEDROCK_PROMPT_VERSION` back to the previous number. `DRAFT` is the working copy:
handy while editing, avoid it in production. Every `/chat` turn (`audit.llm`) and brief run
(`audit.brief`) logs `prompt_name`, `prompt_version` and `prompt_hash` (sha256, first 16 hex of the
stripped text), and stored briefs carry `prompt: {name, version, hash}`.

## IAM

The credentials used by the app need `bedrock:GetPrompt` on the prompt ARN(s):

```json
{
  "Effect": "Allow",
  "Action": "bedrock:GetPrompt",
  "Resource": [
    "arn:aws:bedrock:<region>:<account-id>:prompt/<BEDROCK_PROMPT_CHAT_ID>",
    "arn:aws:bedrock:<region>:<account-id>:prompt/<BEDROCK_PROMPT_BRIEF_ID>"
  ]
}
```

(`bedrock:InvokeModel` is still needed for the chat model.)

## Runtime behaviour

- Fetched lazily on first use (the API starts even if AWS is unreachable) with `GetPrompt`, using
  the `TEXT` template of the `defaultVariant` (else the first variant), and cached per
  (id, version) for `PROMPT_CACHE_TTL_SECONDS` (default 300). The cache holds the raw template;
  `{{var}}` placeholders (Bedrock convention) are rendered on every call with
  `PromptVersion.render(**variables)`. The audit `prompt_hash` identifies the raw template.
- If a refresh fails, the last good text keeps being served and a warning is logged
  (stale-if-error); a failed first fetch returns an error.
- Missing id: 503 `PROMPT_ID_MISSING`. Other errors: `PROMPT_NOT_FOUND`, `PROMPT_ACCESS_DENIED`,
  `PROMPT_UNAVAILABLE`, `PROMPT_INVALID`, `PROMPT_VARIABLES_MISSING` (see `README.md`).
- The prompt is read through the GetPrompt API, not passed as a prompt ARN to Converse, because
  Converse does not allow `system` / `toolConfig` / `inferenceConfig` together with a managed
  prompt, and the chat needs `toolConfig`.
