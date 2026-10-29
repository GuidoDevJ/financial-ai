# Internal pilot checklist (Phase 1, Week 4)

What must be true before the internal pilot, how each acceptance criterion is verified, and which
steps are still manual. Commands are PowerShell-friendly.

## 1. Environment variables

Required for a pilot (`APP_ENV` other than `dev` refuses to run unauthenticated):

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_ENV` | `dev` | Use `pilot`/`prod`. Anything but `dev` requires `AUTH_API_KEYS` (503 `AUTH_NOT_CONFIGURED` otherwise) |
| `AUTH_API_KEYS` | unset | JSON `{"<api key>": "<cliente_id>"}`, e.g. `{"k1...": "client_001"}`. Secret. Generate keys with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ADMIN_API_KEYS` | unset | Comma-separated or JSON list of admin keys. Secret. Required outside `APP_ENV=dev` for `POST /news/ingest` (503 `ADMIN_NOT_CONFIGURED` otherwise); use keys different from `AUTH_API_KEYS` |
| `AWS_REGION`, `BEDROCK_MODEL_ID` | `us-east-1`, `us.anthropic.claude-sonnet-5-5` | |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | unset | Or an IAM role (default boto3 chain). Prefer a role/temporary credentials |
| `BEDROCK_PROMPT_CHAT_ID`, `BEDROCK_PROMPT_BRIEF_ID` | empty | Prompt Management ids/ARNs |
| `BEDROCK_PROMPT_VERSION` | `DRAFT` | `DRAFT` or an integer `1..99999`. Any other value (for example the typo `DRIFT`) aborts startup with a clear message. Use a numbered version for the pilot |
| `BANK_NAME` | `Banco Comafi` | `{{banco}}` prompt variable |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis Stack (RediSearch) for news and checkpoints |
| `IOL_USER`, `IOL_PASSWORD` | empty | BYMA prices; without them BYMA falls back to yfinance `.BA` |
| `NEWS_RSS_URLS`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL_ID` | see README | News RAG |
| `INGEST_BATCH_SIZE`, `INGEST_MAX_RETRIES`, `INGEST_RETRY_BASE_SECONDS` | `10`, `3`, `1.0` | Ingest batching and retry/backoff (optional) |
| `NEWS_USER_AGENT`, `NEWS_MAX_FEED_BYTES` | `financial-ai-news-ingest/1.0`, `5000000` | Feed fetch hygiene (optional) |
| `ENABLE_SCHEDULER` | `false` | Exactly one instance should enable it (news ingest + 07:30 weekday briefs) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO`, `SMTP_STARTTLS` | empty / `587` / `true` | Simulated delivery to one test mailbox |

Week 4 settings (all optional, with defaults):

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHAT_TIMEOUT_SECONDS` | `25` | Whole-turn deadline; exceeded -> 504 `CHAT_TIMEOUT` |
| `CHAT_MAX_TOOL_ITERATIONS` | `6` | Model->tools rounds per turn before a graceful message (`stop_reason: tool_loop_limit`) |
| `PRICE_STALE_TTL_SECONDS` | `604800` | Last-known price kept this long and served flagged `stale` when every provider fails |
| `MACRO_STALE_TTL_SECONDS` | `604800` | Same for macro variables |
| `TASA_BCRA_STALE_TTL_SECONDS` | `2592000` | Same for `tasa_bcra` (30 days: it changes slowly) |

## 2. IAM permissions

For the identity the API runs as (role preferred over access keys):

| Action | Resource | Used by |
| --- | --- | --- |
| `bedrock:InvokeModel` | the chat model / inference profile (`BEDROCK_MODEL_ID`, plus the foundation models behind a cross-region profile) and the embedding model | `/chat`, brief generation, news embeddings. Converse calls are authorized by this action |
| `bedrock:InvokeModelWithResponseStream` | same | Streaming (`ConverseStream`); not used yet, include it so enabling streaming needs no IAM change |
| `bedrock:GetPrompt` | the two prompt ARNs | Prompt Management (`app.prompts.check` and runtime) |
| `kms:Decrypt` | the prompts' customer-managed key, only if one is used | Prompt Management |

`bedrock:ListPrompts` is **not** needed: the app addresses prompts by id. Model access must also be
enabled in the Bedrock console for the region.

## 3. Endpoints and authentication

All of `/chat`, `/brief/preview`, `/brief` and `/brief/latest` take `X-API-Key`. The key selects
the client; a `cliente_id` in the body (or query for `GET /brief/latest`) is optional and must
match it (403 `CLIENT_MISMATCH`). Missing or wrong key: 401. The chat tools `get_portfolio` and
`get_last_brief` read the client from the authenticated session, never from model arguments
(`cliente_id_override_attempt` WARNING if the model tries). `/health` is public.

Every non-2xx body is `{"error": {"code", "message", "request_id"}}` and the id is also in the
`X-Request-ID` header and in every audit line of the request.

## 4. Acceptance criteria -> evidence

| Criterion | Offline evidence (runs in `uv run pytest`) | Live / manual evidence |
| --- | --- | --- |
| 100% of quantitative data comes from a tool call | `tests/test_traceability.py` (numbers in the final reply must appear in tool outputs or the user message; otherwise an `audit.tools.chat` `untraced_numbers` WARNING); brief number validation in `tests/test_brief_validate.py` | During the pilot, search the logs for `untraced_numbers` and review each hit; `tool_calls` in every `/chat` response and `audit.tools` lines show the data path |
| Rejects explicit recommendation requests (adversarial set) | `tests/test_adversarial_recommendations.py` over `tests/data/adversarial_recommendations.json` (38 questions: direct, disguised, role-play, instruction override, authority, urgency, English, comparisons). Compliant replies pass; explicit buy/sell replies are replaced by a safe refusal and flagged `recommendation_blocked` | `uv run python scripts/eval_adversarial.py --url ... --api-key ...` against the real model. It reports pass rate (server guardrail included) and the model-native rate (prompt alone). Pilot gate: pass rate 100%, and review the model-native rate to improve the prompt |
| Brief generated for at least 3 example portfolios | `tests/test_brief_scheduler.py::test_acceptance_three_mock_portfolios_generate_without_error` | `foreach ($c in "client_001","client_002","client_003") { uv run python -m app.brief.generate --cliente $c --store }` or `POST /brief` per client |
| Daily brief without errors on every weekday for a week | `tests/test_brief_week.py` (five consecutive weekdays x three clients; cron fires Monday-Friday 07:30 Buenos Aires) | **Pending**: run the scheduler (`ENABLE_SCHEDULER=true`) for a real week and check `audit.brief` lines (`status: ok`) plus `GET /brief/latest` each morning |
| Chat p95 below 5 s | `tests/test_latency.py` (100 calls, scripted LLM, mocked sources with delays; application overhead only) | `uv run python scripts/bench_chat.py --url ... --api-key ... --n 100`; p50/p95/p99 are printed and the exit code is 1 when p95 >= 5 s. `audit.chat` `duration_ms` has the real distribution during the pilot |
| Compares the portfolio against the market/brief but never recommends | The `compare-*` and `english-04` items of the adversarial set (neutral comparisons pass, comparisons with advice are blocked) | Same `eval_adversarial.py` run; spot-check with real portfolios |

## 5. Failure behaviour to verify once against real services

| Situation | Expected |
| --- | --- |
| IOL down / no credentials | BYMA price comes from yfinance `.BA` (`fuente: "yfinance (.BA)"`), `audit.tools.price` `fallback` WARNING with `from`/`to`/`reason` |
| All price/macro sources down, a previous value exists | The value is returned with `stale: true`, `as_of`, `cached_at`, `stale_reason`; `stale_served` WARNING |
| All sources down, no previous value | Tool answers `{"error": "PROVIDER_UNAVAILABLE", "message": ...}`; the assistant says the data is unavailable |
| `tasa_bcra` primary endpoint down | Secondary BCRA route, then the last known value (30 days) flagged stale |
| Redis down | `/health` reports `degraded`; chat works with in-memory conversation state (`checkpointer_fallback` WARNING) and uncached tools; `/brief` and `/brief/latest` answer 503 `STORAGE_UNAVAILABLE` |
| Concurrent `POST /brief` for the same client and date | 409 `BRIEF_IN_PROGRESS` |
| Slow turn | 504 `CHAT_TIMEOUT` after `CHAT_TIMEOUT_SECONDS` |

## 6. Manual steps still pending

1. Create and publish both prompts in Bedrock Prompt Management (see `docs/prompts/README.md`),
   set the numeric `BEDROCK_PROMPT_VERSION`, run `uv run python -m app.prompts.check`.
2. Generate and distribute `AUTH_API_KEYS` (one per pilot user or team, mapped to its client);
   set `APP_ENV` to a non-`dev` value. Keys are secrets: keep them in the secret store, rotate by
   editing the JSON and restarting. TLS termination and rate limiting are not part of the app.
   Also generate `ADMIN_API_KEYS` for operators. On-demand news ingestion (global, not per
   client): `curl -X POST https://<host>/news/ingest -H "X-API-Key: <admin key>"` returns the
   counts (`indexed`, `skipped_old`, `skipped_unchanged`, `skipped_invalid`, `purged`, `failed_feeds`, per-feed `feeds`); 409 `INGEST_IN_PROGRESS` means the
   scheduler or another operator holds `lock:ingest_news`. News keys in Redis are
   `news/vector_<sha256(url)[:32]>` under the index `idx:news`.
   What is indexed right now (counts only): `curl "https://<host>/news/stats" -H "X-API-Key: <admin key>"`
   or `uv run python -m app.rag.stats`; check `newest_age_hours` (freshness), `expiring_24h`,
   `untagged`, `index.exists` and `index.num_docs` (raw: `redis-cli FT.INFO idx:news`).
3. Update `.env` by hand with the Week 4 variables above (`.env.example` is stale; this project
   never edits `.env*` files automatically).
4. Validate the unverified integrations with real credentials: IOL token/history endpoints
   (`TODO(unverified)` in `app/tools/get_price.py`), the macro fallback endpoints and the BCRA v2.0
   route (`TODO(unverified)` in `app/tools/get_macro.py`).
5. Run `scripts/eval_adversarial.py` and `scripts/bench_chat.py` against the pilot deployment and
   attach the output to the pilot record.
6. Run the scheduler for a real week (see the table above); decide whether to add an Argentine
   holiday calendar (the job also runs on national holidays).
7. Point `SMTP_*` at a real test mailbox (or keep delivery simulated) and confirm one
   `POST /brief` with `"send": true` arrives.
8. Ship the `audit.*` loggers to the central log store and set alerts on
   `recommendation_blocked`, `cliente_id_override_attempt`, `untraced_numbers`, `stale_served` and
   5xx rates.
9. Redis Stack with persistence for the pilot (briefs are stored there).

## 7. Known gaps

- Streaming responses are not implemented.
- The recommendation guardrail is phrase-based (Spanish and English); a novel phrasing can slip
  through, which is why the live adversarial evaluation and log review are part of the pilot.
- `untraced_numbers` is an audit signal, not a filter: replies are not blocked for it. Single
  digit integers are ignored (list numbering).
- If Redis dies after startup, the Redis checkpointer is not swapped at runtime (a restart picks
  the in-memory fallback); tool caches and `/health` do degrade gracefully.
- API keys are static and held in an environment variable; there is no user management or
  per-key rate limiting.
