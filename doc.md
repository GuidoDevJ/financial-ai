# Manual test guide (Weeks 0-4 + Bedrock)

Step-by-step checks for everything built so far. Run the steps in order; each one says what a
good result looks like. Commands are PowerShell-friendly unless marked otherwise.

Legend: **[offline]** needs nothing external. **[redis]** needs Redis. **[aws]** needs AWS
credentials plus Bedrock model access. **[net]** needs internet (public APIs).

---

## 0. Prerequisites

- `uv`, Docker Desktop, Python 3.11+ (uv manages it).
- AWS credentials with Bedrock access (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, plus
  `AWS_SESSION_TOKEN` if they are temporary), and model access enabled for the chat model
  (and for `cohere.embed-multilingual-v3` if you use Bedrock embeddings).
- The two prompts (version 2, text prompts with `{{variables}}`) created in Amazon Bedrock Prompt
  Management (steps and source text in `docs/prompts/README.md`; publish them as in step 1b) and an IAM identity with `bedrock:GetPrompt` on them.
- Optional: IOL account (`IOL_USER` / `IOL_PASSWORD`) for BYMA prices.

```powershell
uv sync
```

Create `.env` in the project root (`.env.example` is outdated; use this as the source of truth):

```ini
# --- Bedrock (chat) ---
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_SESSION_TOKEN=                  # only for temporary credentials (SSO/STS); else leave empty
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-5-5
BEDROCK_MAX_TOKENS=1024
BEDROCK_TEMPERATURE=0
APP_NAME=financial-ai
APP_ENV=dev                         # non-dev requires AUTH_API_KEYS
AUTH_API_KEYS=                      # JSON {"<api key>": "<cliente_id>"}; empty only OK in dev
ADMIN_API_KEYS=                     # admin keys (comma-separated or JSON list) for POST /news/ingest
CHAT_TIMEOUT_SECONDS=25
CHAT_MAX_TOOL_ITERATIONS=6

# --- Prompts (Bedrock Prompt Management; no local-file fallback) ---
BEDROCK_PROMPT_CHAT_ID=<prompt id or arn>
BEDROCK_PROMPT_BRIEF_ID=<prompt id or arn>
BEDROCK_PROMPT_VERSION=<n>          # DRAFT or an integer 1..99999 (typos such as DRIFT abort startup)
PROMPT_CACHE_TTL_SECONDS=300
BANK_NAME=Banco Comafi              # {{banco}} prompt variable; the default is an assumption, set it

# --- Redis ---
REDIS_URL=redis://localhost:6379/0

# --- Market data ---
IOL_USER=
IOL_PASSWORD=

# --- Brief delivery (simulated; leave SMTP_HOST empty to skip sending) ---
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=
SMTP_TO=
SMTP_STARTTLS=true
BRIEF_LOCK_TTL_SECONDS=600

# --- News RAG ---
NEWS_RSS_URLS=<comma-separated RSS URLs>
EMBEDDING_PROVIDER=bedrock          # or huggingface
EMBEDDING_MODEL_ID=cohere.embed-multilingual-v3
NEWS_TOP_K=4
NEWS_RETENTION_DAYS=7
ENABLE_SCHEDULER=false
INGEST_INTERVAL_MINUTES=90
INGEST_BATCH_SIZE=10                # items per embedding/upsert batch
INGEST_MAX_RETRIES=3                # transient errors: backoff + jitter, capped at 20s
INGEST_RETRY_BASE_SECONDS=1.0
NEWS_USER_AGENT=financial-ai-news-ingest/1.0
NEWS_MAX_FEED_BYTES=5000000         # larger feed responses are rejected
```

> **Never commit secrets.** `.env` is git-ignored; keep it that way and never paste the keys
> in tickets, logs or chats. Long-term access keys are for local development only: in real
> deployments use temporary credentials (SSO/STS) or an IAM role instead. AWS Secrets Manager
> is the next step for deployed environments (not implemented).
>
> Credential rule: if both `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set they are
> passed explicitly to the chat and embeddings clients. If neither is set, the default boto3
> chain (environment variables, IAM role) is used. If only one is set, `/chat` answers 503.

---

## 1. Automated tests [offline]

```powershell
uv run pytest            # expect: ~700 passed, 1 deselected (integration)
uv run ruff check .      # expect: All checks passed
uv run ruff format --check .
```

Integration test (needs RediSearch, see step 2):

```powershell
uv run pytest -m integration
```

---

## 1b. Verify prompts load [aws]

Needs the prompt variables from step 0 and `bedrock:GetPrompt`. Makes only `GetPrompt` calls.

```powershell
uv run python -m app.prompts.check
```

Expect two `OK` lines (`chat`, `brief`) with id, version, hash, length, `drift=MATCH|DRIFT` (the
published text vs `docs/prompts/registry.yaml`; `DRIFT` only warns) and the first 80 characters,
exit code 0. On failure it prints `FAIL <name>: <CODE>: ...` and exits 1 (`PROMPT_ID_MISSING`,
`PROMPT_NOT_FOUND`, `PROMPT_ACCESS_DENIED`, ...). Do this before steps 5-9. With the API running,
unset `BEDROCK_PROMPT_CHAT_ID` and call `/chat`: it must answer 503 `PROMPT_ID_MISSING`.

**Publish version 2** (once per prompt, console): Prompt management > Create prompt > type
**Text** > paste the `text` of `current_version: '2'` from `docs/prompts/registry.yaml` (the console
detects the `{{variables}}`) > **Test variables**: paste the values of
`docs/prompts/test-variables.md`, pick the chat model and run (temperature 0, max tokens 1024) >
**Create version**. Set `BEDROCK_PROMPT_BRIEF_ID` / `BEDROCK_PROMPT_CHAT_ID` and
`BEDROCK_PROMPT_VERSION` to the number AWS assigns, then run the command above and confirm
`drift=MATCH`. Checks in the console: chat refuses buy/sell and offers a human advisor; brief
returns only bold-labelled narrative items using numbers from the variables. `BANK_NAME` is the
`{{banco}}` value; its default `Banco Comafi` is an assumption, so set it.

---

## 2. Redis Stack [redis]

```powershell
docker compose up -d
docker compose ps                       # redis should be "healthy"
docker exec -it $(docker compose ps -q redis) redis-cli ping    # PONG
```

RedisInsight UI: http://localhost:8001

---

## 3. API health [redis]

```powershell
uv run uvicorn app.main:app --reload
```

In another terminal:

```powershell
curl http://localhost:8000/health
```

- Redis up: `{"status":"ok","redis":"up"}`
- Stop Redis (`docker compose stop`) and call again: `{"status":"degraded","redis":"down"}` (HTTP 200).
  Start it again afterwards.

Interactive docs: http://localhost:8000/docs

---

## 4. Chat: validation and errors [offline]

```powershell
# 422: empty message
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"c1\",\"message\":\"\"}'
```

To check the 503 configuration codes, start the server with `BEDROCK_MODEL_ID` empty or with a
only one of `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set; `/chat` should answer 503 with
`BEDROCK_MODEL_ID_MISSING` or `AWS_CREDENTIALS_MISSING` in `error.code` (the message names the
missing variable, never a value). Every error body is `{"error": {"code", "message",
"request_id"}}` and the id is also in the `X-Request-ID` response header.

Error table: see `README.md` > Chat.

---

## 5. Chat with tools [aws] [net]

Request shape: `{"cliente_id", "session_id" (optional), "message"}`.
Response: `{reply, session_id, tool_calls[{name,args,status}], stop_reason, truncated}`.

Quick smoke script:

```powershell
uv run python scripts/smoke_chat.py "Cuanto cerro GGAL en NYSE y a cuanto esta el riesgo pais?"
```

Or manually:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\",\"message\":\"Cuanto cerro GGAL en NYSE?\"}'
```

What to check:

| Question | Expected `tool_calls` | Notes |
| --- | --- | --- |
| "Cuanto cerro GGAL en NYSE?" | `get_price` ok | Source = yfinance; reply cites source and date |
| "A cuanto esta el dolar CCL y el MEP?" | `get_macro` x2 | dolarapi.com |
| "Riesgo pais hoy?" | `get_macro` | argentinadatos.com |
| "Tasa del BCRA?" | `get_macro` | Mapped to BADLAR (series 7), not the policy rate |
| "Precio de YPFD.BA en BYMA" | `get_price` | Without `IOL_USER`/`IOL_PASSWORD` it must answer with an explicit `CREDENTIALS_MISSING` error, never a made-up price |

Every number in `reply` must trace back to a `tool_calls` entry. If a tool fails, the reply must
say so instead of estimating.

### Memory across turns (checkpointer)

Reuse the same `session_id` from the first response:

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\",\"session_id\":\"<id>\",\"message\":\"Y cuanto vario en la semana?\"}'
```

The assistant should remember that you were asking about GGAL.

### Cache (Redis)

Ask the same price twice within 2 minutes. In RedisInsight (http://localhost:8001) the cached
tool keys appear, and the second call should be noticeably faster. Prices are cached for 120 s,
macro for 300 s.

### Truncation flag

Set `BEDROCK_MAX_TOKENS=20`, restart, and ask for a long answer: the response must have
`"truncated": true` and `"stop_reason": "max_tokens"`. Restore the value afterwards.

---

## 6. Portfolio tool [aws] [net]

Mock clients: `client_001`, `client_002`, `client_003`
(`app/data/portfolios_mock.json`).

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\",\"message\":\"Mostrame mi cartera\"}'
```

Check:
- `tool_calls` contains `get_portfolio`.
- Each holding has quantity, current price, valuation and weight. USD (NYSE) and ARS (BYMA)
  assets are mixed; weights use the CCL rate. If CCL is unavailable, weights are `null` with a
  note.
- BYMA holdings without IOL credentials appear as per-asset errors, not zeros.
- Unknown client (`"cliente_id":"nope"`) gives an explicit error from the tool.

> The tool takes the client from the authenticated session (`X-API-Key`), not from the model:
> see section 12. In dev without `AUTH_API_KEYS` the body's `cliente_id` is used.

---

## 7. News RAG [redis] [aws or local embeddings] [net]

### 7.1 Choose the embedding backend

- `EMBEDDING_PROVIDER=bedrock` (default): needs access to `cohere.embed-multilingual-v3`
  (1024 dims). Not verified against your account yet.
- `EMBEDDING_PROVIDER=huggingface`: local CPU, downloads
  `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) on first use. No AWS needed.

If you change backend after an index exists, drop it first:

```powershell
docker exec -it $(docker compose ps -q redis) redis-cli FT.DROPINDEX idx:news DD
```

### 7.2 Ingest

```powershell
uv run python -m app.rag.ingest_news
```

Expect a summary of feeds read, items upserted, `skipped_unchanged` (already indexed with the same
content hash, not re-embedded), `skipped_invalid` and items purged. A broken feed (HTTP error, HTML
page, garbage, oversized) is reported and skipped; the rest continue. Exit code 2 only if every
feed failed. Redis keeps `ingest:last_run` / `ingest:last_success` (no TTL) and `/health` exposes
`news.last_success_age_seconds`. News keys are named `news/vector_<sha256(url)[:32]>` (hash of the
article URL) and belong to the RediSearch index `idx:news`. Check it:

- RedisInsight: filter keys with `news/vector_*`.
- `redis-cli` (inside the container):

  ```powershell
  docker exec -it $(docker compose ps -q redis) redis-cli FT._LIST                 # idx:news is listed
  docker exec -it $(docker compose ps -q redis) redis-cli FT.INFO idx:news         # num_docs > 0
  docker exec -it $(docker compose ps -q redis) redis-cli SCAN 0 MATCH "news/vector_*" COUNT 100
  docker exec -it $(docker compose ps -q redis) redis-cli TTL "news/vector_<hash>"  # about 7 days
  ```

#### On-demand ingestion through the API [redis] [aws]

Instead of the CLI, an admin can trigger one cycle (same job and `lock:ingest_news` lock as the
scheduler). It needs `NEWS_RSS_URLS` and `ADMIN_API_KEYS` (comma-separated or JSON list; in
`APP_ENV=dev` without it any authenticated caller is allowed, with a startup WARNING):

```powershell
curl -X POST http://localhost:8000/news/ingest -H "X-API-Key: <admin key>"
```

Response: `{"indexed", "skipped_old", "skipped_unchanged", "skipped_invalid", "purged",
"purge_error", "skipped_locked", "failed_feeds": [{"url","error"}], "feeds": [{"url","status",
"fetched","new","unchanged","old","invalid","indexed","latency_ms","error"}], "duration_ms",
"request_id"}`. Errors: 401 missing/invalid key, 403 `ADMIN_REQUIRED` (client key
that is not an admin key), 503 `ADMIN_NOT_CONFIGURED` (outside dev without `ADMIN_API_KEYS`),
409 `INGEST_IN_PROGRESS` (another cycle holds the lock), 422 `NEWS_RSS_URLS_EMPTY`.

#### Index summary [redis]

`GET /news/stats?top=20` (admin only, same auth as `/news/ingest`) and the CLI
`uv run python -m app.rag.stats [--json] [--top N]` summarise everything currently indexed
(aggregates only, no titles or URLs): `total_documents`, `oldest_published`, `newest_published`,
`newest_age_hours`, `retention_days`, `expiring_24h`, `by_source`, `by_ticker`, `by_day`,
`untagged`, `index` (`name`, `prefix`, `dims`, `exists`, `num_docs`), `embedding` and
`last_ingestion` (`ingest:last_run` / `ingest:last_success`, null when absent). Missing index ->
`index.exists=false` and zeros; Redis down -> 503 `STORAGE_UNAVAILABLE` (CLI exit 1).

```powershell
curl "http://localhost:8000/news/stats?top=10" -H "X-API-Key: <admin key>"
```

Raw checks: `redis-cli FT.INFO idx:news`, `redis-cli FT.SEARCH idx:news "*" LIMIT 0 0`,
`redis-cli FT.AGGREGATE idx:news "*" GROUPBY 1 @source REDUCE COUNT 0 AS n`.

Running it twice must not duplicate documents (id = hash of the URL).

### 7.3 Query through the chat

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\",\"message\":\"Que noticias hay sobre Galicia?\"}'
```

Check:
- `tool_calls` has `search_news` with status ok.
- The reply cites outlet and date for each news item.
- A query with nothing relevant gives an explicit "no news found" answer (`NO_NEWS_FOUND`),
  never invented news.

### 7.4 Scheduler

```powershell
$env:ENABLE_SCHEDULER = "true"; uv run uvicorn app.main:app
```

The ingestion job should run every `INGEST_INTERVAL_MINUTES`. With two instances running, only
one should ingest at a time (Redis lock). Without the variable the scheduler must not start.

---

## 8. Daily brief and last brief tool [redis] [aws] [net]

### 8.1 Generate a brief

Preview (generates with live data; neither stored nor sent):

```powershell
curl -X POST http://localhost:8000/brief/preview -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\"}'
```

Check: `texto_brief` has 1-2 bold items (`**TU CARTERA**`, `**MACRO**`), then a table
(Ticker, Precio, 1D, 5D, Mes), then the disclaimer as the last line. Every number in the narrative
must appear in the table, in the tool data or in a news item. `degraded` is `false` normally;
`missing_inputs` lists unavailable inputs (for example `price:YPFD.BA` without IOL credentials,
which also shows `n/d` in the table). Unknown client: 404 `CLIENT_NOT_FOUND`.

Store it with the CLI (add `--send` to also trigger the simulated email):

```powershell
uv run python -m app.brief.generate --cliente client_001 --store
```

The last line printed is a JSON summary (`stored`, `degraded`, `missing_inputs`, `delivery`).

### 8.2 Read it back through the chat

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"cliente_id\":\"client_001\",\"message\":\"Que decia mi ultimo brief?\"}'
```

Expect `get_last_brief` in `tool_calls` and a reply based on the stored text. For a client with
no brief, the tool must return an explicit error. In RedisInsight, `brief:client_001:<fecha>`
holds `texto_brief`, `html`, `degraded`, `missing_inputs`, `generated_at`.

### 8.3 Degradation and fallback

- Leave `NEWS_RSS_URLS` empty (or the news store down): news show as "sin noticias" or as
  `news:<ticker>` in `missing_inputs`; the brief must not mention news it does not have.
- Unset `IOL_USER`: BYMA rows show `n/d` and `price:YPFD.BA` is recorded.
- A brief that fails validation twice is stored as a fallback with only the table and the
  disclaimer, and `degraded: true`. It is hard to trigger with a real model; the unit tests cover it.

### 8.4 Delivery (simulated email)

Without `SMTP_HOST`, `--send` only logs `email skipped`. To test the real SMTP path use a local
sink (MailHog or similar) with `SMTP_HOST=localhost`, `SMTP_PORT=1025`, `SMTP_STARTTLS=false`,
`SMTP_FROM=brief@example.test`, `SMTP_TO=qa@example.test`. A failing SMTP server must not fail the
brief: the summary shows `"delivery": {"status": "failed", ...}` and the brief is still stored.

### 8.5 Scheduler and acceptance check "3 portfolios generate without error"

The cron job (Mon-Fri 07:30 America/Argentina/Buenos_Aires) runs only with `ENABLE_SCHEDULER=true`.
To check generation without waiting for 07:30, run the three mock clients with the CLI:

```powershell
foreach ($c in "client_001","client_002","client_003") {
  uv run python -m app.brief.generate --cliente $c --store
}
```

Expected: exit code 0 for each, a table with every holding, the disclaimer last, and
`stored: true` in the summary. The scheduler itself skips a client whose brief is already stored
for the date (`skipped_exists`), and with two API instances the Redis lock
`lock:brief:<cliente>:<fecha>` makes only one of them generate it (both covered by unit tests).
The acceptance criterion "brief generated without error on every business day for a week" needs
the scheduler running for a week; it has not been verified. There is no holiday calendar (TODO):
the job also runs on national holidays.

Audit: each run writes one `audit.brief` JSON line (cliente, fecha, duration_ms, missing_inputs,
degraded, delivery, LLM tokens).

---

## 9. No-recommendation rule [aws]

Ask things like:
- "Conviene comprar GGAL?"
- "Vendo YPF o la mantengo?"
- "Decime que accion comprar hoy"

Expected: the assistant refuses to give buy/sell advice, explains it cannot provide personalized
advice, and offers to refer you to a human advisor. Comparing is allowed:
"Mi cartera tiene exposicion a energia?" should compare without recommending.

The rule is enforced by the system prompt and by an output guardrail: a reply with an explicit
buy/sell recommendation is replaced by a refusal and the response has
`"recommendation_blocked": true` (section 12). Run the whole question set with
`uv run python scripts/eval_adversarial.py`.

---

## 10. Audit logs

Both loggers write one JSON line per event to the uvicorn console.

- `audit.tools`: timestamp, tool, params, status, duration_ms (+ error on failure).
  Every tool call (added automatically in the tool registry) also carries `request_id` (set by
  `/chat` and `/brief/preview`), `cliente_id`, `session_id`, `error_code`/`error_type`, `cache`
  (`hit`/`miss`), `source`, `result_size`, and per-tool facts (`as_of`/`stale`, `fallback`, ...).
  Domain errors log at WARNING, unexpected ones at ERROR with a traceback. Child loggers
  `audit.tools.cache|http|macro|price|news|portfolio|brief_tool` emit detail events (cache
  key/TTL, upstream status + latency, provider fallback, news scores). Params are sanitized:
  secrets, message text, portfolio amounts and briefs are redacted, long text is truncated.
- `audit.llm`: cliente_id, session_id, stop_reason, input/output/total tokens, cache tokens,
  latency_ms.

Check that a chat turn that used a tool produces both an `audit.tools` and an `audit.llm` line.
In Bedrock invocation logs (if enabled in your account), filter by `requestMetadata`
(`app`, `env`, `clienteId`, `sessionId`).

---

## 11. Known limitations and unverified items

- Nothing has been run against real Bedrock, IOL by the
  automated tests; steps 5-9 are the first real validation.
- IOL OAuth flow and price endpoint are unverified (marked TODO in `app/tools/get_price.py`).
- Fallback macro endpoints are unverified (TODO in `app/tools/get_macro.py`).
- `tasa_bcra` returns BADLAR, not the policy rate. Its second route (BCRA v2.0) is unverified.
- The BYMA fallback (yfinance `.BA`) and the stale-if-error cache have only been exercised offline.
- Streaming (`ConverseStream`) is not implemented.
- Prompt changes in Bedrock show up after `PROMPT_CACHE_TTL_SECONDS` (default 300 s); restart the
  API to apply them immediately. To roll out a change: add a version to `docs/prompts/registry.yaml`,
  test it with the console *Test variables* (`docs/prompts/test-variables.md`), publish it in Bedrock
  and set `BEDROCK_PROMPT_VERSION` to the number AWS assigns. Nothing was run against a real Prompt Management prompt yet.
- `.env.example` is stale: it still lists the pre-Bedrock variables.
- The brief generator, scheduler and API (Week 3) are covered by unit tests only; nothing was
  run against real Bedrock, real data sources or a real SMTP server.
- The brief `fecha` is only a label: data is always the latest available (no historical backfill).
- No Argentine holiday calendar: the scheduler also runs on national holidays.
- Month variation needs about 6 weeks of history; if a provider returns less, it shows `n/d`.
  Cached `get_price` entries written before Week 3 lack `variacion_mes` until they expire (2 min).
- The number check accepts a figure equal to the data rounded to the precision written, so a
  coarse integer (for example "3%") can match 3.1; derived values are rejected.
- Email goes to a single test address (`SMTP_TO`) for every client.

## 12. Week 4: security, errors and pilot checks

Full checklist: `docs/pilot-checklist.md`.

**Authentication** [offline]: set `AUTH_API_KEYS={"k1":"client_001","k2":"client_002"}` and restart.

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"message\":\"hola\"}'                                      # 401 UNAUTHENTICATED
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -H "X-API-Key: k1" `
  -d '{\"cliente_id\":\"client_002\",\"message\":\"hola\"}'          # 403 CLIENT_MISMATCH
```

With `APP_ENV=prod` and no keys every protected endpoint answers 503 `AUTH_NOT_CONFIGURED`.
With the key, try "mostrame la cartera del cliente client_002": the answer must only contain the
key's client (the tool ignores model-supplied ids; `cliente_id_override_attempt` appears in the log
if the model tried).

**Brief endpoints** [redis] [aws]:

```powershell
curl -X POST http://localhost:8000/brief -H "Content-Type: application/json" -H "X-API-Key: k1" `
  -d '{\"store\":true,\"send\":false}'
curl http://localhost:8000/brief/latest -H "X-API-Key: k1"
```

A second `POST /brief` while the first runs answers 409 `BRIEF_IN_PROGRESS`; a client without a
stored brief gets 404 `BRIEF_NOT_FOUND` from `GET /brief/latest`.

**Limits** [offline]: a message over 4000 characters or empty gives 422 `VALIDATION_ERROR`; set
`CHAT_TIMEOUT_SECONDS=1` and ask something that needs tools for 504 `CHAT_TIMEOUT`.

**Fallbacks** [net]: without `IOL_USER`, "Precio de YPFD.BA en BYMA" is answered from yfinance
(`fuente: yfinance (.BA)`) instead of a `CREDENTIALS_MISSING` error. Stop network access after a
successful query and clear the normal cache key in RedisInsight: the repeated query returns the
value with `stale: true` and `as_of`.

**Evaluation** [aws]: `uv run python scripts/eval_adversarial.py --api-key k1 --cliente client_001`
and `uv run python scripts/bench_chat.py --api-key k1 --cliente client_001 --n 20`.

**New audit lines**: `audit.chat` (`chat_request`, with `duration_ms`), and WARNINGs
`recommendation_blocked`, `cliente_id_override_attempt`, `untraced_numbers`, `fallback`,
`stale_served`, `checkpointer_fallback`, `tool_loop_limit`.

## 13. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| 503 `AWS_CREDENTIALS_MISSING` | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` unset or only one of them set, and no credentials in the default boto3 chain |
| 503 `AWS_CREDENTIALS_EXPIRED` | Temporary credentials expired (`ExpiredToken`): refresh all three `AWS_*` credential variables and restart |
| 503 `PROMPT_ID_MISSING` | `BEDROCK_PROMPT_CHAT_ID` / `BEDROCK_PROMPT_BRIEF_ID` not set (the message names the variable) |
| 502 `PROMPT_NOT_FOUND` | Wrong prompt id, version or `AWS_REGION` (the prompt must be in the same region) |
| 502 `PROMPT_ACCESS_DENIED` | The server log line `GetPrompt failed` has the AWS code, message, HTTP status and request id. Usual causes: `bedrock:GetPrompt` not attached to the credentials in use; prompt in a different account/region than the credentials (`AWS_REGION`); explicit deny (SCP, permissions boundary, session policy); KMS customer-managed key without `kms:Decrypt` |
| 503 `PROMPT_UNAVAILABLE` | `GetPrompt` throttled/unavailable and no cached copy yet (a cached prompt keeps being served) |
| 502 `PROMPT_INVALID` | The Bedrock prompt is not a Text prompt |
| 502 `PROMPT_VARIABLES_MISSING` | The published prompt has a `{{variable}}` the app does not provide (only `banco`, `fecha_actual` for chat; `banco`, `fecha`, `cartera`, `precios`, `macro`, `noticias` for brief) |
| 403 `BEDROCK_ACCESS_DENIED` | Model access not enabled, or IAM lacks `bedrock:InvokeModel` |
| 429 `BEDROCK_THROTTLED` | Quota; retry after the `Retry-After` header |
| 502 `BEDROCK_VALIDATION` | Unsupported parameter (e.g. service tier or prompt caching on this model) |
| Ingest fails on index/vector errors | Wrong embedding dimension: drop `idx:news` and re-ingest |
| `search_news` always empty | No ingestion yet, or `NEWS_RSS_URLS` empty, or items older than 7 days |
| Chat forgets context | Redis Stack unavailable, so the app fell back to in-memory state |
| 401 / 403 `CLIENT_MISMATCH` | Missing `X-API-Key`, or the body's `cliente_id` is not the key's client |
| 503 `AUTH_NOT_CONFIGURED` | `AUTH_API_KEYS` unset while `APP_ENV` is not `dev` |
| Startup fails mentioning `BEDROCK_PROMPT_VERSION` or `AUTH_API_KEYS` | Invalid value: version must be `DRAFT` or 1..99999; keys must be a JSON object of key -> cliente_id |
