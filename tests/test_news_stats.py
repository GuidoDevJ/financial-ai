"""News index summary: aggregation, Redis path (fake RediSearch), endpoint and CLI."""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient
from llama_index.core.schema import Document
from llama_index.core.vector_stores import SimpleVectorStore

from app.core.config import Settings
from app.main import app, get_news_summary_job
from app.rag import stats
from app.rag.ingest_news import INGEST_LAST_RUN_KEY, INGEST_LAST_SUCCESS_KEY
from app.rag.redis_store import RedisNewsVectorStore
from app.rag.stats import news_index_summary
from app.rag.store import NewsStore
from tests.fakes import BagOfWordsEmbedding

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
ADMIN = "admin-secret-key"
CLIENT_KEYS = '{"key-one": "c1"}'
SECRET_TITLE = "SECRET-TITLE-XYZ"
SECRET_URL = "https://leak.example/secret-article"


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def item(n: int, source: str, tickers: str, published: datetime) -> dict:
    return {
        "title": f"{SECRET_TITLE} {n}",
        "summary": "s",
        "source": source,
        "url": f"{SECRET_URL}/{n}",
        "published": published.isoformat(),
        "published_epoch": int(published.timestamp()),
        "tickers": tickers,
    }


CUTOFF = NOW - timedelta(days=7)
ITEMS = [
    item(1, "Ambito", "GGAL,YPF", NOW - timedelta(hours=1)),  # newest
    item(2, "Ambito", "GGAL", NOW - timedelta(hours=30)),
    item(3, "Reuters", "", NOW - timedelta(days=2)),  # untagged
    item(4, "Reuters", "YPF", CUTOFF + timedelta(hours=3)),  # expires in 3h
    item(5, "Bloomberg", "", NOW - timedelta(days=5)),  # untagged
    item(6, "Reuters", "", NOW - timedelta(days=8)),  # already stale (awaiting purge)
]


def seeded_store(items=ITEMS) -> NewsStore:
    store = NewsStore(SimpleVectorStore(), BagOfWordsEmbedding(), retention_days=7)
    store.upsert([Document(id_=f"id-{i['url']}", text=i["title"], metadata=i) for i in items])
    return store


# ------------------------------------------------------------ simple store


def test_aggregation_on_seeded_store():
    s = news_index_summary(seeded_store(), settings(), now=NOW)
    assert s["total_documents"] == 6
    assert s["oldest_published"] == (NOW - timedelta(days=8)).isoformat()
    assert s["newest_published"] == (NOW - timedelta(hours=1)).isoformat()
    assert s["newest_age_hours"] == 1.0
    assert s["retention_days"] == 7
    assert s["expiring_24h"] == 1  # the stale one is not "expiring", it is expired
    assert s["by_source"] == {"Reuters": 3, "Ambito": 2, "Bloomberg": 1}
    assert s["by_ticker"] == {"GGAL": 2, "YPF": 2}
    assert s["untagged"] == 3
    days = s["by_day"]
    assert len(days) == 7 and list(days) == sorted(days)
    assert days["2026-10-19"] == 1 and days["2026-10-18"] == 1 and days["2026-10-17"] == 1
    assert days["2026-10-14"] == 1  # the 5-day-old item
    assert sum(days.values()) == 4  # items on days before the 7-day window are not bucketed


def test_empty_store():
    s = news_index_summary(seeded_store([]), settings(), now=NOW)
    assert s["total_documents"] == 0
    assert s["oldest_published"] is None and s["newest_published"] is None
    assert s["newest_age_hours"] is None
    assert s["by_source"] == {} and s["by_ticker"] == {} and s["untagged"] == 0
    assert s["expiring_24h"] == 0 and sum(s["by_day"].values()) == 0
    assert s["index"]["exists"] is True


def test_top_limits_sources_and_tickers():
    s = news_index_summary(seeded_store(), settings(), now=NOW, top=1)
    assert s["by_source"] == {"Reuters": 3}
    assert len(s["by_ticker"]) == 1


def test_health_block_and_embedding_from_settings():
    s = news_index_summary(
        seeded_store(), settings(news_index_name="idx:x", news_index_prefix="px"), now=NOW
    )
    assert s["index"]["name"] == "idx:x" and s["index"]["prefix"] == "px"
    assert s["index"]["dims"] == 1024
    assert s["embedding"] == {
        "provider": "bedrock",
        "model_id": "cohere.embed-multilingual-v3",
    }


def test_last_ingestion_from_redis_and_absent():
    fake = fakeredis.FakeRedis(decode_responses=True)
    s = news_index_summary(seeded_store(), settings(), now=NOW, redis_client=fake)
    assert s["last_ingestion"] == {"last_run": None, "last_success": None}
    fake.set(
        INGEST_LAST_RUN_KEY,
        json.dumps({"finished_at": "2026-10-19T11:00:00+00:00", "indexed": 4, "ok": True}),
    )
    fake.set(INGEST_LAST_SUCCESS_KEY, "2026-10-19T11:00:00+00:00")
    s = news_index_summary(seeded_store(), settings(), now=NOW, redis_client=fake)
    assert s["last_ingestion"]["last_run"]["indexed"] == 4
    assert s["last_ingestion"]["last_success"] == "2026-10-19T11:00:00+00:00"


def test_last_ingestion_is_best_effort():
    class Broken:
        def get(self, key):
            raise redis.ConnectionError("down")

    s = news_index_summary(seeded_store(), settings(), now=NOW, redis_client=Broken())
    assert s["last_ingestion"] == {"last_run": None, "last_success": None}


def test_summary_has_no_titles_or_urls():
    text = json.dumps(news_index_summary(seeded_store(), settings(), now=NOW))
    assert SECRET_TITLE not in text and "leak.example" not in text


# ------------------------------------------------------- fake RediSearch


class FakeFT:
    """Just enough of redis-py's `ft()` to exercise the aggregation requests."""

    def __init__(self, owner):
        self.owner = owner

    def info(self):
        self.owner.calls.append("info")
        if self.owner.fail:
            raise redis.ConnectionError("down redis://:pw@h")
        if self.owner.missing:
            raise redis.ResponseError("Unknown index name")
        return {"num_docs": str(len(self.owner.docs))}

    def aggregate(self, req):
        if self.owner.missing:
            raise redis.ResponseError("no such index")
        args = [str(a) for a in req.build_args()]
        self.owner.calls.append(("aggregate", args))
        g = args.index("GROUPBY")
        n = int(args[g + 1])
        fields = [a.lstrip("@") for a in args[g + 2 : g + 2 + n]]
        docs = self.owner.docs

        def value(doc, name):
            if name == "day":
                return str(doc["published_epoch"] // 86400)
            return str(doc[name]) if doc.get(name) not in (None, "") else None

        if not fields:
            epochs = [d["published_epoch"] for d in docs]
            if not docs:
                return SimpleNamespace(rows=[])
            return SimpleNamespace(
                rows=[
                    [
                        "n",
                        str(len(docs)),
                        "min_epoch",
                        str(min(epochs)),
                        "max_epoch",
                        str(max(epochs)),
                    ]
                ]  # fmt: skip
            )
        counts: dict = {}
        for d in docs:
            v = value(d, fields[0])
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        return SimpleNamespace(rows=[[fields[0], k, "n", str(c)] for k, c in counts.items()])

    def search(self, query):
        if self.owner.missing:
            raise redis.ResponseError("no such index")
        lo, hi = re.search(r"\[(\S+) \((\S+)\]", query.query_string()).groups()
        total = sum(1 for d in self.owner.docs if int(lo) <= d["published_epoch"] < int(hi))
        return SimpleNamespace(total=total, docs=[])


class FakeSearchRedis(fakeredis.FakeRedis):
    def __init__(self, docs, missing=False):
        super().__init__(decode_responses=True)
        self.docs = docs
        self.missing = missing
        self.fail = False
        self.calls: list = []

    def ft(self, index_name="idx"):
        return FakeFT(self)


def redis_store(docs, missing=False) -> tuple[RedisNewsVectorStore, FakeSearchRedis]:
    client = FakeSearchRedis(docs, missing)
    vs = RedisNewsVectorStore(
        index_name="idx:news",
        index_prefix="news",
        dims=1024,
        retention_seconds=7 * 86400,
        redis_client=client,
    )
    return vs, client


def test_redis_path_matches_simple_path():
    vs, client = redis_store(ITEMS)
    expected = news_index_summary(seeded_store(), settings(), now=NOW)
    got = news_index_summary(vs, settings(), now=NOW, redis_client=client)
    assert got["index"]["num_docs"] == 6
    for key in (
        "total_documents", "oldest_published", "newest_published", "newest_age_hours",
        "expiring_24h", "by_source", "by_day", "by_ticker", "untagged",
    ):  # fmt: skip
        assert got[key] == expected[key], key


def test_redis_path_uses_server_side_aggregation_without_loading_text():
    vs, client = redis_store(ITEMS)
    news_index_summary(vs, settings(), now=NOW, redis_client=client)
    aggregates = [c[1] for c in client.calls if isinstance(c, tuple)]
    assert aggregates and all("GROUPBY" in a for a in aggregates)
    flat = " ".join(" ".join(a) for a in aggregates)
    assert "@source" in flat and "@tickers" in flat and "@published_epoch" in flat
    assert "title" not in flat and "url" not in flat


def test_missing_index_reports_not_exists_with_zeros():
    vs, client = redis_store([], missing=True)
    s = news_index_summary(vs, settings(), now=NOW, redis_client=client)
    assert s["index"]["exists"] is False and s["index"]["num_docs"] is None
    assert s["total_documents"] == 0 and s["by_source"] == {} and s["untagged"] == 0
    assert s["newest_published"] is None


def test_empty_redis_index():
    vs, client = redis_store([])
    s = news_index_summary(vs, settings(), now=NOW, redis_client=client)
    assert s["index"]["exists"] is True and s["total_documents"] == 0


def test_redis_errors_propagate():
    vs, client = redis_store(ITEMS)
    client.fail = True
    with pytest.raises(redis.ConnectionError):
        news_index_summary(vs, settings(), now=NOW, redis_client=client)


# ----------------------------------------------------------------- endpoint


def teardown_function():
    app.dependency_overrides.clear()


def client_with(job=None) -> TestClient:
    job = job or (lambda top: news_index_summary(seeded_store(), settings(), now=NOW, top=top))
    app.dependency_overrides[get_news_summary_job] = lambda: job
    return TestClient(app)


def get_stats(client, key=None, **params):
    return client.get("/news/stats", headers={"X-API-Key": key} if key else {}, params=params)


def test_dev_without_keys_is_allowed_and_shape():
    r = get_stats(client_with())
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {
        "total_documents", "oldest_published", "newest_published", "newest_age_hours",
        "retention_days", "expiring_24h", "by_source", "by_day", "by_ticker", "untagged",
        "index", "embedding", "last_ingestion", "request_id",
    }  # fmt: skip
    assert body["total_documents"] == 6
    assert body["request_id"] == r.headers["x-request-id"]
    assert SECRET_TITLE not in r.text and "leak.example" not in r.text


@pytest.mark.parametrize("env", ["prod", "staging"])
def test_outside_dev_without_admin_keys_is_503(set_env, env):
    set_env(app_env=env, auth_api_keys=CLIENT_KEYS)
    r = get_stats(client_with(), "key-one")
    assert r.status_code == 503 and r.json()["error"]["code"] == "ADMIN_NOT_CONFIGURED"


def test_missing_key_is_401(set_env):
    set_env(admin_api_keys=ADMIN)
    r = get_stats(client_with())
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_unknown_key_is_401(set_env):
    set_env(admin_api_keys=ADMIN, auth_api_keys=CLIENT_KEYS)
    r = get_stats(client_with(), "nope")
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_API_KEY"


def test_client_key_is_403(set_env):
    set_env(admin_api_keys=ADMIN, auth_api_keys=CLIENT_KEYS)
    r = get_stats(client_with(), "key-one")
    assert r.status_code == 403 and r.json()["error"]["code"] == "ADMIN_REQUIRED"


def test_admin_key_is_accepted(set_env):
    set_env(admin_api_keys=ADMIN, auth_api_keys=CLIENT_KEYS)
    assert get_stats(client_with(), ADMIN).status_code == 200


def test_top_param_is_applied_and_validated():
    c = client_with()
    assert list(get_stats(c, top=1).json()["by_source"]) == ["Reuters"]
    assert get_stats(c, top=100).status_code == 200
    for bad in (0, 101, -1):
        assert get_stats(c, top=bad).status_code == 422


def test_redis_down_is_503_standard_body():
    def job(top):
        raise redis.ConnectionError("boom redis://:pw@h")

    r = get_stats(client_with(job))
    assert r.status_code == 503 and r.json()["error"]["code"] == "STORAGE_UNAVAILABLE"
    assert "pw@h" not in r.text


def test_runs_off_the_event_loop_thread():
    import threading

    seen = {}

    def job(top):
        seen["t"] = threading.current_thread()
        return news_index_summary(seeded_store(), settings(), now=NOW, top=top)

    get_stats(client_with(job))
    assert seen["t"] is not threading.main_thread()


def test_audit_event_has_counts_only(set_env, caplog):
    set_env(admin_api_keys=ADMIN)
    caplog.set_level(logging.INFO)
    r = get_stats(client_with(), ADMIN)
    assert r.status_code == 200
    lines = [x.getMessage() for x in caplog.records if x.name.startswith("audit.tools.news")]
    event = json.loads(next(line for line in lines if "news_stats" in line))
    assert event["total_documents"] == 6 and event["request_id"] == r.json()["request_id"]
    everything = "\n".join(x.getMessage() for x in caplog.records)
    assert ADMIN not in everything and SECRET_TITLE not in everything
    assert "leak.example" not in everything


# ---------------------------------------------------------------------- CLI


def run_cli(monkeypatch, capsys, argv, job=None):
    job = job or (lambda top: news_index_summary(seeded_store(), settings(), now=NOW, top=top))
    monkeypatch.setattr(stats, "build_summary_job", lambda s=None: job)
    code = stats.main(argv)
    return code, capsys.readouterr()


def test_cli_text_output(monkeypatch, capsys):
    code, out = run_cli(monkeypatch, capsys, [])
    assert code == 0
    assert "Total documents" in out.out and "6" in out.out
    assert "Reuters" in out.out and "GGAL" in out.out and "untagged" in out.out.lower()
    assert "idx:news" in out.out
    assert SECRET_TITLE not in out.out and "leak.example" not in out.out


def test_cli_json_output_and_top(monkeypatch, capsys):
    code, out = run_cli(monkeypatch, capsys, ["--json", "--top", "1"])
    assert code == 0
    data = json.loads(out.out)
    assert data["total_documents"] == 6 and list(data["by_source"]) == ["Reuters"]


def test_cli_redis_unreachable_exits_1(monkeypatch, capsys):
    def job(top):
        raise redis.ConnectionError("boom redis://:pw@h")

    code, out = run_cli(monkeypatch, capsys, ["--json"], job)
    assert code == 1
    assert "pw@h" not in out.out + out.err and out.err.strip()


def test_cli_rejects_bad_top(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, capsys, ["--top", "0"])
    assert e.value.code == 2
