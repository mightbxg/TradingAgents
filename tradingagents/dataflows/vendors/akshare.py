"""Eastmoney news and financial statements through AKShare.

News is a relevance-ranked recent sample, not an archive. Statements without
publication dates are live-only; report periods are not publication dates.
"""

import importlib
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import withhold_undated_statements
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.search_terms import company_search_names, mentions_company
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)
CHINA_TZ = ZoneInfo("Asia/Shanghai")


def _client():
    return importlib.import_module("akshare")


def _symbol(ticker):
    canonical = normalize_symbol(ticker)
    hk = re.fullmatch(r"(\d{1,5})\.HK", canonical)
    if hk:
        return canonical, hk[1].zfill(5), "hk"
    cn = re.fullmatch(r"(\d{6})\.(SS|SZ)", canonical)
    if cn:
        return canonical, cn[1], "sh" if cn[2] == "SS" else "sz"
    raise NoMarketDataError(ticker, canonical, "AKShare adapter supports .HK, .SS and .SZ listings")


def _fetch(method, **kwargs):
    try:
        return getattr(_client(), method)(**kwargs)
    except Exception as exc:
        raise VendorUnavailableError(f"AKShare {method} failed: {type(exc).__name__}") from exc


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    canonical, code, _ = _symbol(ticker)
    names = []
    try:
        profile = get_company_profile(canonical)
        names = [profile.get("longName"), profile.get("shortName")]
    except Exception:
        logger.debug("Company lookup unavailable for %s; using configured aliases", canonical)
    names = company_search_names(canonical, names)
    queries = list(dict.fromkeys([code, *names]))
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=CHINA_TZ)
    stop = min(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=CHINA_TZ)
               + timedelta(days=1), datetime.now(CHINA_TZ))
    articles, seen, failures = [], set(), []
    for query in queries:
        try:
            data = _fetch("stock_news_em", symbol=query)
        except VendorUnavailableError as exc:
            failures.append(str(exc))
            continue
        if data is None or data.empty:
            continue
        required = {"新闻标题", "新闻内容", "发布时间", "新闻链接"}
        if not required <= set(data.columns):
            failures.append("unexpected news columns")
            continue
        for row in data.to_dict("records"):
            title, summary = str(row["新闻标题"]), str(row["新闻内容"])
            if not mentions_company(title + " " + summary, [canonical, code, *names]):
                continue
            published = pd.to_datetime(row["发布时间"], errors="coerce")
            if pd.isna(published):
                continue
            published = (published.tz_localize(CHINA_TZ) if published.tzinfo is None
                         else published.tz_convert(CHINA_TZ))
            if not start <= published < stop:
                continue
            link = row["新闻链接"]
            key = link if isinstance(link, str) and link else (title, published.isoformat())
            if key in seen:
                continue
            seen.add(key)
            articles.append((published, title, summary, str(row.get("文章来源", "Eastmoney")), link))
    if not articles:
        detail = "; ".join(failures) if failures else "no relevant, dated results in the requested window"
        raise VendorUnavailableError(
            f"Eastmoney news search for {canonical}: {detail}. Recent search samples do not "
            "establish an absence of company news"
        )
    articles.sort(key=lambda a: a[0], reverse=True)
    header = (f"## {ticker} News, from Eastmoney via AKShare\n"
              f"Search terms: {', '.join(queries)}\n"
              "Times: Asia/Shanghai. Recent search sample; content is a snippet, not full text.\n")
    if failures:
        header += f"Partial retrieval: {len(failures)} searches failed.\n"
    parts = [header]
    for published, title, summary, source, link in articles[:get_config()["news_article_limit"]]:
        parts.append(f"### {title} (source: {source})\nPublished: {published.isoformat()}\n"
                     f"Snippet: {summary}\nLink: {link}\n")
    return "\n".join(parts)


def _statement(ticker, freq, as_of_date, title, cn_method):
    canonical, code, market = _symbol(ticker)
    if freq.lower() not in {"annual", "quarterly"}:
        raise ValueError("freq must be annual or quarterly")
    withheld = withhold_undated_statements(as_of_date, canonical, title)
    if withheld:
        raise VendorUnavailableError(f"AKShare statement withheld: {withheld}")
    if market == "hk":
        data = _fetch("stock_financial_hk_report_em", stock=code, symbol=title,
                      indicator="年度" if freq.lower() == "annual" else "报告期")
    else:
        data = _fetch(cn_method, symbol=market.upper() + code)
    if data is None or data.empty:
        raise NoMarketDataError(ticker, canonical, f"no AKShare {title} data")
    if "REPORT_DATE" not in data.columns:
        raise VendorUnavailableError("AKShare statement has no report date")
    data = data.copy()
    dates = pd.to_datetime(data["REPORT_DATE"], errors="coerce")
    cutoff = pd.Timestamp(as_of_date or datetime.now(CHINA_TZ).date())
    keep = dates.notna() & (dates <= cutoff)
    if freq.lower() == "annual" and market != "hk":
        keep &= (dates.dt.month == 12) & (dates.dt.day == 31)
    data = data.loc[keep]
    dates = dates.loc[keep]
    periods = dates.drop_duplicates().sort_values(ascending=False).head(4)
    data = data.loc[dates.isin(periods)].copy()
    if data.empty:
        raise NoMarketDataError(ticker, canonical, "no statement rows before analysis date")
    if market == "hk":
        needed = {"STD_ITEM_CODE", "STD_ITEM_NAME", "AMOUNT"}
        if not needed <= set(data.columns):
            raise VendorUnavailableError("AKShare HK statement columns changed")
        columns = [c for c in ("REPORT_DATE", "START_DATE", "STD_ITEM_CODE", "STD_ITEM_NAME", "AMOUNT")
                   if c in data.columns]
        data = data[columns].sort_values("REPORT_DATE", ascending=False)
    header = (f"# {title} for {ticker} ({freq}), Eastmoney via AKShare\n"
              f"Source: https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index?code={code}\n"
              if market == "hk" else f"# {title} for {ticker} ({freq}), Eastmoney via AKShare\n")
    header += (
        "Report dates denote accounting periods, not filing dates. Current vendor snapshot; "
        "historical publication timing and revisions are not verified.\n"
        "Amounts are unscaled vendor values. Use only explicit currency/unit fields in the CSV; "
        "where absent, that metadata is unavailable. Do not infer currency or silently convert amounts.\n"
        "The quarterly option means available report periods, which can include half-year, "
        "annual and cumulative figures; do not label them as standalone quarters.\n"
    )
    return header + data.to_csv(index=False)


def get_balance_sheet(ticker, freq="quarterly", as_of_date=None):
    return _statement(ticker, freq, as_of_date, "资产负债表", "stock_balance_sheet_by_report_em")


def get_income_statement(ticker, freq="quarterly", as_of_date=None):
    return _statement(ticker, freq, as_of_date, "利润表", "stock_profit_sheet_by_report_em")


def get_cashflow(ticker, freq="quarterly", as_of_date=None):
    return _statement(ticker, freq, as_of_date, "现金流量表", "stock_cash_flow_sheet_by_report_em")
