# financial-ai

Investment-information assistant (LLM + market APIs) for Argentine equities. All market data
comes from tools, never from the model. See the Phase 1 spec PDF for details.

## Setup

```bash
uv sync
cp .env.example .env   # set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / BEDROCK_MODEL_ID, etc.
```

### Prompts (Amazon Bedrock Prompt Management)

All system prompts live in Bedrock Prompt Management; the code reads them by ID and never from
local files. Same convention as the sibling project `ia-comafi-infra`: one env var per prompt id
(`BEDROCK_PROMPT_<NAME>_ID`) and one global version. `docs/prompts/registry.yaml` is the
hand-maintained source of truth / changelog (not read at runtime). Create the two prompts once
(manually, console or exported CLI payload: see `docs/prompts/README.md`), then set:

```ini
BEDROCK_PROMPT_CHAT_ID=<id or arn>
BEDROCK_PROMPT_BRIEF_ID=<id or arn>
BEDROCK_PROMPT_VERSION=<n>       # number AWS assigns on "Create version" of the v2 text; default DRAFT
PROMPT_CACHE_TTL_SECONDS=300     # in-memory cache; stale text is served if a refresh fails
BANK_NAME=Banco Comafi           # value of the {{banco}} prompt variable (default is an assumption)
```

The prompts are Bedrock **text prompts** that use `{{variables}}` for everything dynamic: `chat`
uses `banco` and `fecha_actual`; `brief` uses `banco`, `fecha`, `cartera`, `precios`, `macro` and
`noticias` (the resolved data, rendered as compact JSON inside the prompt; the user turn is just
`Generá el brief.`). The app renders them on every call (strict: an unknown `{{variable}}` is a
`PROMPT_VARIABLES_MISSING` error; values are inserted verbatim). To publish version 2: create each
prompt as **Text**, paste the text from `registry.yaml` (`current_version: '2'`), try it with the
console *Test variables* (copy-paste values in `docs/prompts/test-variables.md`), **Create version**,
and set `BEDROCK_PROMPT_VERSION` to the number AWS assigns.

The IAM identity needs `bedrock:GetPrompt` on both prompt ARNs. Verify with
`uv run python -m app.prompts.check` (prints name, id, version, hash, length, drift status vs
`registry.yaml` and the first 80 characters; exit 1 only if a prompt cannot be fetched).
`uv run python -m app.prompts.export` prints the `aws bedrock-agent create-prompt` payload offline.
Prompts are fetched lazily on first use, so the API starts without AWS. Every `/chat` turn
(`audit.llm`) and brief run (`audit.brief`) logs `prompt_name`, `prompt_version`, `prompt_hash`;
stored briefs carry `prompt: {name, version, hash}`.

## Redis

```bash
docker compose up -d   # Redis Stack: 6379 (server), 8001 (RedisInsight)
```

## Run

```bash
uv run uvicorn app.main:app --reload
# GET http://localhost:8000/health
```

## Test and lint

```bash
uv run pytest        # uses fakeredis; no Docker needed
uv run ruff check .
```
## Chat (`POST /chat`)

Tool-calling assistant (LangGraph + Claude) using `get_price` and `get_macro`.

```bash
# .env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (AWS_SESSION_TOKEN), AWS_REGION, BEDROCK_MODEL_ID
#       (IOL_USER / IOL_PASSWORD only needed for BYMA prices)
docker compose up -d                  # Redis (cache + checkpointer; Redis Stack for checkpoints)
uv run uvicorn app.main:app --reload
uv run python scripts/smoke_chat.py "Cuanto cerro GGAL en NYSE?"
```

Request: `{"cliente_id": "c1", "session_id": "optional", "message": "..."}` plus the `X-API-Key`
header (see Week 4; `cliente_id` is optional once a key identifies the client). `message` is
trimmed, stripped of control characters, and must be 1-4000 characters (422 otherwise).
Response: `{"reply", "session_id", "tool_calls": [{"name", "args", "status"}], "stop_reason",
"truncated", "recommendation_blocked"}`. `truncated` is `true` when Bedrock stopped with `max_tokens` (raise
`BEDROCK_MAX_TOKENS`); `stop_reason` is the Converse `stopReason` (`end_turn`, `tool_use`,
`max_tokens`, ...).

Errors (every non-2xx body is `{"error": {"code", "message", "request_id"}}`; the table lists `error.code`):

| HTTP | Code | Cause |
| --- | --- | --- |
| 422 | `VALIDATION_ERROR` | Invalid body (empty/oversized message, bad `fecha`, ...) |
| 401 / 403 / 503 | `UNAUTHENTICATED`, `INVALID_API_KEY` / `CLIENT_MISMATCH` / `AUTH_NOT_CONFIGURED` | Authentication (Week 4) |
| 504 | `CHAT_TIMEOUT` | The turn exceeded `CHAT_TIMEOUT_SECONDS` |
| 503 | `BEDROCK_MODEL_ID_MISSING`, `AWS_REGION_MISSING`, `AWS_CREDENTIALS_MISSING` | Local configuration |
| 503 | `AWS_CREDENTIALS_EXPIRED` | `ExpiredToken`: refresh the temporary credentials |
| 403 | `BEDROCK_ACCESS_DENIED` | `AccessDeniedException`: enable model access / IAM `bedrock:InvokeModel` |
| 429 | `BEDROCK_THROTTLED` | `ThrottlingException`; sends `Retry-After: 5` |
| 502 | `BEDROCK_VALIDATION` | `ValidationException` (bad request, e.g. unsupported parameter) |
| 503 | `BEDROCK_UNAVAILABLE` | `ModelNotReadyException`, `ServiceUnavailableException` |
| 504 | `BEDROCK_TIMEOUT` | `ModelTimeoutException` |
| 503 | `PROMPT_ID_MISSING` | `BEDROCK_PROMPT_CHAT_ID` (or `BEDROCK_PROMPT_BRIEF_ID` on `/brief/preview`) is not set |
| 502 | `PROMPT_NOT_FOUND` | `GetPrompt` `ResourceNotFoundException`: wrong id/version/region |
| 502 | `PROMPT_ACCESS_DENIED` | `GetPrompt` `AccessDeniedException`: AWS detail is in the server log (`GetPrompt failed`); check `bedrock:GetPrompt` on the credentials in use, account/region (`AWS_REGION`), explicit deny (SCP/boundary/session policy), KMS `kms:Decrypt` |
| 503 | `PROMPT_UNAVAILABLE` | `GetPrompt` throttled, service error or unreachable (no cached copy yet) |
| 502 | `PROMPT_INVALID` | The prompt has no TEXT template |
| 502 | `PROMPT_VARIABLES_MISSING` | The published prompt uses a `{{variable}}` the app does not provide |
| 502 | `LLM_ERROR` | Any other failure |

LLM: Amazon Bedrock through the Converse API (`ChatBedrockConverse` inside LangGraph; 30s read
timeout, 3 retries). The system prompt (fetched from Prompt Management) is sent as the Converse `system` field, tools as
`toolConfig`, and `BEDROCK_MAX_TOKENS` / `BEDROCK_TEMPERATURE` as `inferenceConfig`.
Settings: `AWS_REGION` (default `us-east-1`), `BEDROCK_MODEL_ID` (model id or cross-region
inference profile, default `us.anthropic.claude-sonnet-5-5`).

AWS credentials (shared by chat and Bedrock embeddings):

```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=        # only for temporary credentials
```

If both key id and secret are set they are passed explicitly to the clients. If neither is set,
the default boto3 chain (env vars, IAM role) is used and `/chat` answers 503
`AWS_CREDENTIALS_MISSING` only if it finds nothing. If only one is set, it also answers 503.

Warning: never commit secrets (`.env` is git-ignored). Long-term access keys are for local
development; in real deployments use temporary credentials (SSO/STS) or an IAM role. AWS
Secrets Manager is the next step for deployed environments (not implemented).

### Bedrock settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `BEDROCK_MAX_TOKENS` | `1024` | Converse `inferenceConfig.maxTokens` |
| `BEDROCK_TEMPERATURE` | `0` | Converse `inferenceConfig.temperature` |
| `APP_NAME` / `APP_ENV` | `financial-ai` / `dev` | Tags sent in Converse `requestMetadata` |

Each model call also sends `requestMetadata` (`app`, `env`, `clienteId`, `sessionId`; nothing
else personal), which you can filter on in Bedrock model invocation logs.

The no-recommendation rule is enforced by the system prompt and, on the way out, by a
deterministic output guardrail (Week 4).

### Tool audit log

Each tool call writes one `audit.tools` JSON line with `request_id`, `cliente_id`, `session_id`, sanitized
`params`, `status`, `error_code`, `duration_ms`, `cache`, `source` and `result_size`; detail events go to
`audit.tools.<cache|http|macro|price|news|portfolio|brief_tool>`. Secrets, message text and portfolio
contents are never logged (see `sanitize` in `app/core/audit.py`).

### LLM audit log

Besides the `audit.tools` logger, the `audit.llm` logger emits one JSON line per model response:
`timestamp`, `event`, `cliente_id`, `session_id`, `stop_reason`, `input_tokens` (uncached),
`output_tokens`, `total_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens`,
`latency_ms` (Converse `metrics.latencyMs`).

Not implemented: streaming (`ConverseStream` / SSE `POST /chat/stream`).
If the Redis checkpointer is unavailable the app falls back to in-memory conversation state.

## Week 2: news RAG, portfolio and brief tools

The chat now has five tools: `get_price`, `get_macro`, `search_news`, `get_portfolio`,
`get_last_brief`. News is stored as one LlamaIndex document per item (title + lede) in a
`VectorStoreIndex` over Redis (RediSearch, Redis Stack required), with a 7-day retention.

### New environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEWS_RSS_URLS` | empty | Comma-separated RSS feed URLs to ingest |
| `NEWS_RETENTION_DAYS` | `7` | Per-key TTL and purge window |
| `NEWS_TOP_K` | `4` | Results returned by `search_news` |
| `NEWS_HTTP_TIMEOUT` | `10` | Seconds per feed request |
| `NEWS_USER_AGENT` | `financial-ai-news-ingest/1.0` | `User-Agent` sent when fetching feeds |
| `NEWS_MAX_FEED_BYTES` | `5000000` | Max feed response size (streamed; larger = feed failure) |
| `INGEST_BATCH_SIZE` | `10` | Items embedded/upserted per batch (a failing batch does not lose the feed) |
| `INGEST_MAX_RETRIES` | `3` | Retries for transient embedding/fetch errors (exponential backoff + jitter, cap 20s, honors `Retry-After`) |
| `INGEST_RETRY_BASE_SECONDS` | `1.0` | Backoff base |
| `NEWS_INDEX_NAME` / `NEWS_INDEX_PREFIX` | `idx:news` / `news` | RediSearch index and key prefix; keys are `news/vector_<sha256(url)[:32]>` |
| `EMBEDDING_PROVIDER` | `bedrock` | `bedrock` or `huggingface` (local CPU) |
| `EMBEDDING_MODEL_ID` | `cohere.embed-multilingual-v3` | Bedrock model id; with `huggingface` and the default value, `paraphrase-multilingual-MiniLM-L12-v2` is used |
| `EMBEDDING_DIMENSIONS` | derived | Index vector size (1024 for Cohere v3, 384 for MiniLM). Required for other models |
| `ENABLE_SCHEDULER` | `false` | Start the ingestion job inside the API process |
| `INGEST_INTERVAL_MINUTES` | `90` | Scheduler interval |
| `INGEST_LOCK_TTL_SECONDS` | `900` | Redis lock (`SET NX EX`) so only one instance ingests |
| `BEDROCK_PROMPT_CHAT_ID` | empty | Bedrock Prompt Management prompt id/ARN for `/chat` (required) |
| `BEDROCK_PROMPT_BRIEF_ID` | empty | Same, for the daily brief |
| `BEDROCK_PROMPT_VERSION` | `DRAFT` | Version read for every prompt (`DRAFT` or a number) |
| `PROMPT_CACHE_TTL_SECONDS` | `300` | In-memory prompt cache TTL |
| `BANK_NAME` | `Banco Comafi` | Bank name sent as the `{{banco}}` prompt variable (the default is an assumption: set it) |

Bedrock embeddings reuse `AWS_REGION` and the same `AWS_*` credentials, and need `bedrock:InvokeModel` on the
embedding model (request model access in the Bedrock console). The index dimension is fixed
when the index is created: after changing the embedding model, drop the index
(`FT.DROPINDEX idx:news DD`) and re-ingest.

### Ingest

```bash
docker compose up -d                        # Redis Stack
# .env: NEWS_RSS_URLS=https://feed1,https://feed2
uv run python -m app.rag.ingest_news        # one cycle: fetch, upsert by URL hash, purge > 7 days
ENABLE_SCHEDULER=true uv run uvicorn app.main:app   # or run it periodically in the API process
```

An admin can also trigger one cycle over HTTP (same job and `lock:ingest_news` lock):

```bash
curl -X POST http://localhost:8000/news/ingest -H "X-API-Key: <admin key>"
# {"indexed":12,"skipped_old":3,"skipped_unchanged":40,"skipped_invalid":1,"purged":0,"purge_error":null,
#  "skipped_locked":false,"failed_feeds":[],"feeds":[{"url":"https://feed1/rss","status":"ok","fetched":20,
#  "new":12,"unchanged":8,"old":0,"invalid":0,"indexed":12,"latency_ms":310.2,"error":null}],"duration_ms":4210.5,"request_id":"..."}
```

### Index summary (what is indexed now, not the last run)

Admin-only aggregate view of the whole news index: counts only, never titles, summaries or URLs.

```bash
curl "http://localhost:8000/news/stats?top=20" -H "X-API-Key: <admin key>"   # top 1..100, default 20
uv run python -m app.rag.stats [--json] [--top N]                            # exit 1 if Redis is unreachable
```

Fields: `total_documents`; `oldest_published` / `newest_published` (UTC ISO) and `newest_age_hours`
(stale feeds show up here); `retention_days` and `expiring_24h` (items that reach the retention
cutoff within 24h); `by_source` and `by_ticker` (top N, from the `source` / `tickers` tags);
`by_day` (UTC date -> count, last `retention_days` days, zeros included); `untagged` (no ticker);
`index` (`name`, `prefix`, `dims` from settings, `exists`, `num_docs` from `FT.INFO`); `embedding`
(`provider`, `model_id`); `last_ingestion` (`last_run`, `last_success` from the `ingest:last_run` /
`ingest:last_success` keys, null when absent). A missing index answers `index.exists=false` with
zeros. Errors: 401/403 like `/news/ingest`, 503 `STORAGE_UNAVAILABLE` when Redis is down.

Raw equivalents: `redis-cli FT.INFO idx:news` (`num_docs`), `redis-cli FT.SEARCH idx:news "*" LIMIT 0 0`
(total), `redis-cli FT.AGGREGATE idx:news "*" GROUPBY 1 @source REDUCE COUNT 0 AS n`.

Items already indexed with the same `content_hash` (sha256 of title + summary) are skipped, so reruns
do not re-embed. An HTML page, garbage or oversized response counts as a failed feed (`status`
`failed`); a valid feed with no entries is `empty` (WARNING `feed_empty`). The CLI exits 2 only when
every feed failed. Each run writes `ingest:last_run` (and `ingest:last_success`) in Redis without TTL;
`/health` shows `news.last_success_age_seconds` once a run has succeeded.

It needs `ADMIN_API_KEYS` (comma-separated or JSON list; in `APP_ENV=dev` without it any
authenticated caller is allowed, with a startup WARNING; elsewhere 503 `ADMIN_NOT_CONFIGURED`).
A client key that is not an admin key gets 403 `ADMIN_REQUIRED`; a running cycle gives 409
`INGEST_IN_PROGRESS`; empty `NEWS_RSS_URLS` gives 422 `NEWS_RSS_URLS_EMPTY`. The ingestion runs in a
worker thread and failed-feed errors are sanitized and truncated.

To check the data in RedisInsight or `redis-cli`: keys are named `news/vector_<sha256(url)[:32]>` under
the index `idx:news`.

```bash
redis-cli FT._LIST                          # idx:news
redis-cli FT.INFO idx:news                  # num_docs
redis-cli SCAN 0 MATCH "news/vector_*" COUNT 100
```

A failing feed is reported and skipped; the rest are still ingested. Items without a date or
link are dropped. The HTML scraping fallback (`app/rag/scrape_fallback.py`) is only a stub.

### Query

Ask the chat about news (`POST /chat`, e.g. "Que noticias hay sobre Galicia?"); the agent calls
`search_news(query, ticker?, horas=72)` and returns `{titulo, resumen, fuente, url, fecha}` items.
No matches gives an explicit `NO_NEWS_FOUND` error, never invented news.

`get_portfolio` reads `app/data/portfolios_mock.json` (`client_001..003`) and prices each asset
with `get_price`; a failing price is reported on that asset. Mixed USD/ARS weights use the CCL
rate. `get_last_brief` reads briefs saved with `app.brief.store.BriefStore`
(`brief:{cliente_id}:{fecha}`); briefs are generated by the Week 3 job below.

### Tests

```bash
uv run pytest                      # unit tests, no network/AWS/Docker
uv run pytest -m integration       # needs Redis with RediSearch (REDIS_URL)
```


## Week 3: daily brief

A LangGraph pipeline run by the job (never by the LLM):

```
load_portfolio -> [prices | macro | news] (parallel) -> generate -> validate
    -> (retry once | fallback) -> assemble -> store -> send
```

- `get_portfolio` for the client, then `get_price` for every holding, `get_macro`
  (`dolar_ccl`, `dolar_mep`, `riesgo_pais`) and `search_news` (last 24h, per held ticker) are
  called by the job. The generation prompt (Bedrock prompt `BEDROCK_PROMPT_BRIEF_ID`, Spanish; source text
  in `docs/prompts/registry.yaml`) receives the
  resolved data through its `{{cartera}}`, `{{precios}}`, `{{macro}}` and `{{noticias}}` variables
  (compact JSON inside the system prompt); the user turn is the fixed trigger `Generá el brief.`;
  the model has no tools bound.
- Any tool failure degrades gracefully: the item is passed to the prompt as `"no disponible"`, the
  table shows `n/d`, and the run records it in `missing_inputs` (`price:YPFD.BA`,
  `macro:riesgo_pais`, `news:GGAL`). An empty news search (`NO_NEWS_FOUND`) is a valid result, not
  a missing input. BYMA prices need IOL credentials; without them those rows are `n/d`.
- Variations come from the price history: `get_price` now also returns `variacion_mes` (30-day
  lookback; the history window was widened to 2 months / 45 days). 1D = `variacion_dia`,
  5D = `variacion_semana` (7 calendar days). `null` is shown as `n/d`.

### Design: the LLM writes only the narrative

The model writes 1-2 bold-labelled items (`**TU CARTERA**`, `**MACRO**`). The price table and the
fixed disclaimer are rendered in code (`app/brief/render.py`) from the resolved data, so a number in
the table can never be hallucinated and the disclaimer is always present and last.

Deterministic validation (`app/brief/validate.py`, no LLM):

| Rule | Detail |
| --- | --- |
| Numbers traceable | Every number in the narrative must equal some number in the input data (including numbers in news text and dates), rounded to the precision written. es-AR (`1,2%`, `$1.612`) and `1.2` are both accepted; the sign is ignored. Derived values (sums, differences) are rejected |
| No recommendations | Spanish phrase list: `comprá`, `vendé`, `conviene comprar/vender`, `te recomiendo`, `momento de comprar`, `deberías vender`, ... Neutral uses ("la compra de dólares") pass |
| Structure and size | 1-2 items with an upper-case bold label, 40-700 chars and at most 4 lines each; no table or disclaimer written by the model |
| Final text | The disclaimer appears exactly once, at the end; at most 5000 chars |

On failure the generation is retried once with the violation list appended. If it still fails (or
the LLM is unreachable in a scheduled run) a fallback brief is stored with only the deterministic
table and the disclaimer, flagged `degraded: true`.

Stored payload (`brief:{cliente_id}:{fecha}`): `fecha`, `texto_brief`, `activos_mencionados`
(all holdings, as in the table) plus optional `html`, `degraded`, `missing_inputs`,
`generated_at`. `get_last_brief` still returns only the three original fields. The HTML is
self-contained (inline CSS, everything escaped).

### CLI and endpoint

```bash
uv run python -m app.brief.generate --cliente client_001                       # print only
uv run python -m app.brief.generate --cliente client_001 --fecha 2026-10-15 --store --send
```

`--store` saves in Redis (overwrites) and `--send` sends the simulated email. Exit codes: 0 ok,
1 unknown client / invalid input, 2 other failure (including a missing or unreadable prompt). `fecha` only labels the brief: prices, macro and
news are always the latest available (no historical backfill).

`POST /brief/preview` with `{"cliente_id": "client_001", "fecha": "2026-10-15"}` (`fecha`
optional) generates without storing or sending and returns `{fecha, texto_brief, html,
activos_mencionados, degraded, missing_inputs}`. 404 `CLIENT_NOT_FOUND` for an unknown client, 422
for a bad body, and the same Bedrock error mapping as `/chat` (see the table above).

### Scheduler and delivery

With `ENABLE_SCHEDULER=true` (default off) the API process also registers a cron job: Monday to
Friday at 07:30 `America/Argentina/Buenos_Aires`, for `client_001..003` (read from
`app/data/portfolios_mock.json`). Each client is isolated (one failure does not stop the others),
guarded by a Redis lock per client and date (`lock:brief:{cliente_id}:{fecha}`, `SET NX EX`) and
skipped if the brief is already stored, so several instances generate it once. In scheduled runs
an LLM outage produces a degraded data-only brief instead of no brief.

TODO: there is no Argentine holiday calendar; the job also runs on national holidays.

Delivery is simulated: with `SMTP_HOST` unset nothing is sent and `email skipped` is logged. With
it set, the brief is sent as text + HTML via `smtplib` to the single test address `SMTP_TO`. A
delivery failure never fails the brief (it is already stored); its status goes to the audit log.

### New environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMTP_HOST` | empty | Unset = email skipped (simulated) |
| `SMTP_PORT` | `587` | |
| `SMTP_USER` / `SMTP_PASSWORD` | empty | Login is attempted only if both are set (`SMTP_PASSWORD` is a secret) |
| `SMTP_FROM` / `SMTP_TO` | empty | Sender and the single test recipient; both required to send |
| `SMTP_STARTTLS` | `true` | Call `STARTTLS` after connecting |
| `BRIEF_LOCK_TTL_SECONDS` | `600` | TTL of the per-client/date generation lock |

### Brief audit log

The `audit.brief` logger emits one JSON line per run: `event` (`brief_run`), `status`, `cliente_id`,
`fecha`, `duration_ms`, `missing_inputs`, `degraded`, `attempts`, `stored`, `delivery`
(`sent`/`skipped`/`failed`/`not_requested`), `llm` (`calls`, `input_tokens`, `output_tokens`) and
`llm_error`. Each model call also emits the usual `audit.llm` line.


## Week 4: errors, security, audit and pilot readiness

See `docs/pilot-checklist.md` for the full pilot checklist (env vars, IAM, acceptance mapping,
pending manual steps).

### Authentication and client binding

`AUTH_API_KEYS` is a JSON object `{"<api key>": "<cliente_id>"}` (secret). `/chat`,
`/brief/preview`, `/brief` and `/brief/latest` require `X-API-Key` (compared with
`hmac.compare_digest`): missing/invalid -> 401, a `cliente_id` in the body that differs from the
key's client -> 403 `CLIENT_MISMATCH`. When the variable is unset, `APP_ENV=dev` trusts the body's
`cliente_id` (startup WARNING); any other `APP_ENV` answers 503 `AUTH_NOT_CONFIGURED`.

The tools `get_portfolio` and `get_last_brief` no longer accept `cliente_id` from the model: it is
removed from their schemas and taken from the authenticated request context. A value supplied by
the model is ignored and logged (`cliente_id_override_attempt`, WARNING).

### Brief endpoints

- `POST /brief` `{cliente_id?, fecha?, store=true, send=false}`: same pipeline as the CLI and the
  daily job (`run_brief`). Guarded by the scheduler's per-client/date Redis lock (409
  `BRIEF_IN_PROGRESS`). Returns `{cliente_id, fecha, texto_brief, html, activos_mencionados,
  degraded, missing_inputs, stored, sent, delivery_status, generated_at, prompt, request_id}`.
  A stored brief for the same date is overwritten (like the CLI `--store`).
- `GET /brief/latest`: the latest stored brief of the authenticated client (404 `BRIEF_NOT_FOUND`).
- `POST /brief/preview` is unchanged (never stores or sends) apart from authentication.

### Output guardrail and audit

- Final chat replies are checked for explicit buy/sell recommendations (Spanish conjugations and
  imperatives, disguised and English phrasings; refusals such as "no puedo decirte que te conviene
  comprar" are allowed). A violating reply is replaced by a fixed refusal before it reaches the
  client or the conversation history, `recommendation_blocked` is `true` and an
  `audit.tools.guardrail` WARNING `recommendation_blocked` is logged.
- `audit.chat` logs one `chat_request` line per turn: `request_id`, `cliente_id`, `session_id`,
  `status` (`ok`/`error`/`timeout`), `duration_ms`, tool names, `stop_reason`, guardrail flag,
  `untraced_numbers` (count of numbers in the reply that appear in no tool output; details in
  `audit.tools.chat`). The message text is never logged.
- A turn stops after `CHAT_MAX_TOOL_ITERATIONS` tool rounds with a graceful message
  (`stop_reason: tool_loop_limit`).

### Fallbacks

- BYMA prices: IOL first, then yfinance with the `.BA` suffix (`fallback` WARNING with
  `from`/`to`/`reason`).
- `tasa_bcra`: primary BCRA v4 route, then the BCRA v2 route (unverified), then the last known
  value.
- Price and macro: when every provider fails, the last good value (kept for
  `PRICE_STALE_TTL_SECONDS` / `MACRO_STALE_TTL_SECONDS` / `TASA_BCRA_STALE_TTL_SECONDS`) is returned
  with `stale: true`, `as_of`, `cached_at` and `stale_reason`. With no previous value the tool
  returns `{"error": code, "message": ...}` and the assistant must say the data is unavailable.

### Evaluation scripts (opt-in, need a running server and AWS)

```bash
uv run python scripts/eval_adversarial.py --url http://localhost:8000 --api-key KEY   # pass rate
uv run python scripts/bench_chat.py --url http://localhost:8000 --api-key KEY --n 100  # p50/p95/p99
```

The offline counterparts run in `uv run pytest`: `tests/test_adversarial_recommendations.py`
(dataset `tests/data/adversarial_recommendations.json`) and `tests/test_latency.py`.

### New environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `AUTH_API_KEYS` | unset | JSON `{"<api key>": "<cliente_id>"}`; required outside `APP_ENV=dev` |
| `ADMIN_API_KEYS` | unset | Comma-separated or JSON list of admin keys for `POST /news/ingest` and `GET /news/stats`; required outside `APP_ENV=dev` (503 `ADMIN_NOT_CONFIGURED`) |
| `CHAT_TIMEOUT_SECONDS` | `25` | Whole-turn deadline (504 `CHAT_TIMEOUT`) |
| `CHAT_MAX_TOOL_ITERATIONS` | `6` | Tool rounds per turn |
| `PRICE_STALE_TTL_SECONDS` | `604800` | Stale-if-error copy of prices |
| `MACRO_STALE_TTL_SECONDS` | `604800` | Stale-if-error copy of macro values |
| `TASA_BCRA_STALE_TTL_SECONDS` | `2592000` | Stale-if-error copy of `tasa_bcra` |

`BEDROCK_PROMPT_VERSION` is now validated at startup: `DRAFT` or an integer 1..99999.
