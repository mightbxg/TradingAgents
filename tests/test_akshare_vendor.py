"""AKShare routing, company search and point-in-time boundaries."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import akshare as vendor


@pytest.fixture
def client(monkeypatch):
    api = SimpleNamespace(stock_news_em=Mock(), stock_financial_hk_report_em=Mock(),
                          stock_cash_flow_sheet_by_report_em=Mock())
    monkeypatch.setattr(vendor, "_client", lambda: api)
    monkeypatch.setattr(vendor, "get_company_profile", lambda _: {"longName": "Xiaomi Corporation"})
    with run_config({"company_search_aliases": {"1810.HK": ["小米集团", "小米"]}}):
        yield api


def news(title="小米集团业绩", published="2026-10-08 10:00:00", link="https://example.com/news"):
    return {"新闻标题": title, "新闻内容": "小米集团发布消息", "发布时间": published,
            "新闻链接": link, "文章来源": "测试来源"}


def test_news_aliases_dates_dedup_and_partial_failures(client):
    def fetch(symbol):
        if symbol == "小米":
            raise OSError("timeout")
        return pd.DataFrame([
            news(), news(), news("FUTURE", "2026-10-10 10:00:00", "future"),
            news("UNDATED", "bad-date", "undated"),
            {**news("OTHER", link="other"), "新闻内容": "无关公司"},
        ])
    client.stock_news_em.side_effect = fetch
    with run_config({"company_search_aliases": {"1810.HK": ["小米", "小米集团"]}}):
        out = vendor.get_news("1810.HK", "2026-10-02", "2026-10-09")
    terms = [call.kwargs["symbol"] for call in client.stock_news_em.call_args_list]
    assert terms[0] == "01810" and {"Xiaomi", "小米", "小米集团"} <= set(terms)
    assert out.count("### 小米集团业绩") == 1
    assert "FUTURE" not in out and "UNDATED" not in out and "OTHER" not in out
    assert "Partial retrieval" in out and "+08:00" in out and "snippet" in out


def test_china_dates_do_not_shift_into_previous_utc_day(client):
    client.stock_news_em.return_value = pd.DataFrame([
        news("小米EARLY", "2026-10-08 00:30:00", "early"),
        news("小米AFTER", "2026-10-09 00:00:00", "after"),
    ])
    out = vendor.get_news("1810.HK", "2026-10-08", "2026-10-08")
    assert "EARLY" in out and "AFTER" not in out


def test_live_news_is_cut_at_fetch_time(client, monkeypatch):
    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 9, 16, 22, tzinfo=tz)
    monkeypatch.setattr(vendor, "datetime", FixedTime)
    client.stock_news_em.return_value = pd.DataFrame([
        news("小米BEFORE", "2026-10-09 16:20:00", "before"),
        news("小米LATER", "2026-10-09 17:19:00", "later"),
    ])
    out = vendor.get_news("1810.HK", "2026-10-09", "2026-10-09")
    assert "BEFORE" in out and "LATER" not in out


def test_empty_search_is_unavailable_not_silence(client):
    client.stock_news_em.return_value = pd.DataFrame()
    with pytest.raises(VendorUnavailableError, match="absence"):
        vendor.get_news("1810.HK", "2026-10-02", "2026-10-09")


def test_profile_failure_keeps_configured_aliases(client, monkeypatch):
    monkeypatch.setattr(vendor, "get_company_profile", Mock(side_effect=OSError()))
    client.stock_news_em.return_value = pd.DataFrame([news()])
    with run_config({"company_search_aliases": {"1810.HK": ["小米集团"]}}):
        assert "小米集团业绩" in vendor.get_news("1810.HK", "2026-10-02", "2026-10-09")


@pytest.mark.parametrize("ticker,expected", [("1810.HK", "01810"), ("700.HK", "00700"),
                                            ("600519.SH", "600519"), ("000001.SZ", "000001")])
def test_vendor_symbol_mapping(ticker, expected):
    assert vendor._symbol(ticker)[1] == expected


def test_unsupported_market_never_calls_akshare(client):
    with pytest.raises(NoMarketDataError):
        vendor.get_news("AAPL", "2026-10-02", "2026-10-09")
    client.stock_news_em.assert_not_called()


@pytest.mark.parametrize("method,title", [(vendor.get_balance_sheet, "资产负债表"),
                                          (vendor.get_cashflow, "现金流量表"),
                                          (vendor.get_income_statement, "利润表")])
def test_hk_statements_preserve_period_and_amount(client, method, title):
    client.stock_financial_hk_report_em.return_value = pd.DataFrame([
        {"REPORT_DATE": "2026-06-30", "START_DATE": "2026-01-01", "STD_ITEM_CODE": "1",
         "STD_ITEM_NAME": "经营产生现金", "AMOUNT": 8660216000},
        {"REPORT_DATE": "2099-06-30", "START_DATE": "2099-01-01", "STD_ITEM_CODE": "1",
         "STD_ITEM_NAME": "未来", "AMOUNT": 999},
    ])
    out = method("1810.HK")
    client.stock_financial_hk_report_em.assert_called_once_with(stock="01810", symbol=title,
                                                             indicator="报告期")
    assert "8660216000" in out and "2026-01-01" in out and "2099" not in out
    assert "not filing dates" in out and "standalone quarters" in out and "unscaled" in out


def test_historical_statement_does_not_fetch_current_data(client):
    with pytest.raises(VendorUnavailableError, match="withheld"):
        vendor.get_cashflow("1810.HK", as_of_date="2020-01-01")
    client.stock_financial_hk_report_em.assert_not_called()


def test_a_share_annual_cashflow_filters_periods(client):
    client.stock_cash_flow_sheet_by_report_em.return_value = pd.DataFrame([
        {"REPORT_DATE": "2025-12-31", "NETCASH_OPERATE": 12},
        {"REPORT_DATE": "2026-06-30", "NETCASH_OPERATE": 6},
    ])
    out = vendor.get_cashflow("600519.SS", "annual")
    client.stock_cash_flow_sheet_by_report_em.assert_called_once_with(symbol="SH600519")
    assert "2025-12-31" in out and "2026-06-30" not in out


def test_empty_and_failed_statements_are_distinguished(client):
    client.stock_financial_hk_report_em.return_value = pd.DataFrame()
    with pytest.raises(NoMarketDataError):
        vendor.get_cashflow("1810.HK")
    client.stock_financial_hk_report_em.side_effect = OSError()
    with pytest.raises(VendorUnavailableError):
        vendor.get_cashflow("1810.HK")


def test_router_falls_back_to_yahoo_for_unsupported_news(monkeypatch, client):
    yahoo = Mock(return_value="YAHOO NEWS")
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", yahoo)
    with run_config({"data_vendors": {"news_data": "akshare,yfinance"}}):
        assert router.route_to_vendor("get_news", "AAPL", "2026-10-02", "2026-10-09") == "YAHOO NEWS"
    yahoo.assert_called_once()
    client.stock_news_em.assert_not_called()
