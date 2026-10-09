"""Company-name retrieval must work without ticker tags and remain date-safe."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.search_terms import company_search_names, mentions_company
from tradingagents.dataflows.vendors import reddit
from tradingagents.dataflows.vendors.yahoo import news


@pytest.fixture(autouse=True)
def configured_aliases():
    with run_config({"company_search_aliases": {"1810.HK": ["Xiaomi", "小米", "小米集团"]}}):
        yield


def article(title, day, link):
    return {"title": title, "link": link,
            "providerPublishTime": int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())}


def test_aliases_are_configurable_and_not_substring_matches():
    with run_config({"company_search_aliases": {"TEST.HK": ["测试公司"]}}):
        names = company_search_names("TEST.HK", ["Example Corporation", None])
    assert names == ["Example Corporation", "Example", "测试公司"]
    assert mentions_company("测试公司发布财报", names)
    assert not mentions_company("Examples are unrelated", names)
    assert not mentions_company("disappear", ["SAP"])


@pytest.mark.parametrize("title", ["Xiaomi earnings", "小米发布财报"])
def test_untagged_company_news_is_found_and_deduplicated(monkeypatch, title):
    calls = []
    relevant = article(title, "2026-10-08", "company")
    future = article("Xiaomi FUTURE", "2026-10-10", "future")
    unrelated = article("Unrelated earnings", "2026-10-08", "other")
    monkeypatch.setattr(news.yf, "Ticker", lambda _: SimpleNamespace(get_news=lambda **kw: []))

    def search(query, **kwargs):
        calls.append(query)
        return SimpleNamespace(
            quotes=[{"symbol": "1810.HK", "longname": "Xiaomi Corporation"},
                    {"symbol": "OTHER", "longname": "Unrelated"}],
            news=[] if query == "1810.HK" else [relevant, relevant, future, unrelated],
        )

    monkeypatch.setattr(news.yf, "Search", search)
    out = news.get_news_yfinance("1810.HK", "2026-10-02", "2026-10-09")
    assert {"1810.HK", "Xiaomi Corporation", "Xiaomi", "小米"} <= set(calls)
    assert "Unrelated" not in calls
    assert out.count("### " + title) == 1
    assert "FUTURE" not in out and "Unrelated earnings" not in out


def test_generic_company_names_come_from_matching_quote(monkeypatch):
    calls = []
    monkeypatch.setattr(news.yf, "Ticker", lambda _: SimpleNamespace(get_news=lambda **kw: []))

    def search(query, **kwargs):
        calls.append(query)
        return SimpleNamespace(quotes=[{"symbol": "TEST.HK", "longname": "Example Corporation"}],
                               news=[article("Example earnings", "2026-10-08", "example")])

    monkeypatch.setattr(news.yf, "Search", search)
    assert "Example earnings" in news.get_news_yfinance("TEST.HK", "2026-10-02", "2026-10-09")
    assert calls == ["TEST.HK", "Example Corporation", "Example"]


def test_failed_alias_search_preserves_other_results(monkeypatch):
    def search(query, **kwargs):
        if query == "小米":
            raise VendorUnavailableError("outage")
        return SimpleNamespace(quotes=[], news=[article("Xiaomi earnings", "2026-10-08", "ok")])

    monkeypatch.setattr(news.yf, "Search", search)
    found, answered = news._search_news("1810.HK", 20)
    assert answered and len(found) == 1


def test_reddit_uses_one_or_query_and_keeps_date_filter(monkeypatch):
    queries = []

    def fetch(query, *args):
        queries.append(query)
        return [{"title": "小米 earnings", "subreddit": "stocks",
                 "created_utc": datetime(2026, 10, 8, tzinfo=UTC).timestamp()},
                {"title": "Xiaomi FUTURE", "subreddit": "stocks",
                 "created_utc": datetime(2026, 10, 10, tzinfo=UTC).timestamp()}]

    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", fetch)
    out = reddit.fetch_reddit_posts("1810.HK", ("stocks",),
                                   company_names=["Xiaomi Corporation"],
                                   start_date="2026-10-02", end_date="2026-10-09")
    assert len(queries) == 1
    assert '"1810.HK" OR "Xiaomi Corporation" OR "Xiaomi"' in queries[0]
    assert '"小米"' in queries[0]
    assert "小米 earnings" in out and "FUTURE" not in out
