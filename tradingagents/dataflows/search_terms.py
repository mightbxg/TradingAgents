"""Deterministic company aliases for text search, independent of LLM guesses."""

import re

from tradingagents.dataflows.config import get_config


def company_search_names(ticker: str, names=()) -> list[str]:
    """Combine vendor names with explicit, locally configured aliases."""
    aliases = get_config().get("company_search_aliases", {}).get(ticker.upper(), [])
    if isinstance(aliases, str):
        aliases = [aliases]
    result = []
    for name in [*names, *aliases]:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if len(name) < 2:
            continue
        # Remove legal suffixes, not meaningful words in the company's name.
        short = re.sub(
            r"(?:\s+(?:corporation|corp\.?|inc\.?|limited|ltd\.?|plc))+$", "", name,
            flags=re.IGNORECASE,
        ).strip()
        for value in (name, short):
            if len(value) >= 2 and value.casefold() not in {s.casefold() for s in result}:
                result.append(value)
    return result


def mentions_company(text: str, names: list[str]) -> bool:
    """Match aliases as words, so e.g. SAP does not match 'disappear'."""
    return any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I)
               if n.isascii() else n.casefold() in text.casefold() for n in names)
