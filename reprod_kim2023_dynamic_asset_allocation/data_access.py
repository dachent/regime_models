from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PAPER_START = "1970-01-01"
PAPER_END = "2020-12-31"


@dataclass(frozen=True)
class RawModelData:
    growth_raw: pd.DataFrame
    cpi: pd.Series
    uig: pd.Series
    asset_returns: pd.DataFrame
    risk_free: pd.Series
    vix_reference: pd.Series
    notes: list[str]


def _ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _fred_cache_path(cache_dir: Path, series_id: str, start: str, end: str, extras: dict[str, str] | None = None) -> Path:
    parts = [series_id, start, end]
    if extras:
        parts.extend(f"{key}-{value}" for key, value in sorted(extras.items()))
    safe_name = "_".join(part.replace("/", "-") for part in parts)
    return cache_dir / "fred" / f"{safe_name}.csv"


def _yahoo_cache_path(cache_dir: Path, symbol: str, start: str, end: str, interval: str) -> Path:
    safe_symbol = symbol.replace("^", "caret_").replace("/", "_")
    return cache_dir / "yahoo" / f"{safe_symbol}_{start}_{end}_{interval}.csv"


def _fetch_fred_payload(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "regime_models/kk2023"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read()
        if payload.lstrip().startswith(b"<"):
            raise RuntimeError("FRED returned HTML instead of CSV.")
        return payload
    except (HTTPError, URLError, TimeoutError, RuntimeError, OSError):
        completed = subprocess.run(
            ["curl", "-sS", "--retry", "2", "--max-time", "45", url],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FRED curl fallback failed: {stderr}")
        payload = completed.stdout
        if payload.lstrip().startswith(b"<") or not payload.strip():
            raise RuntimeError("FRED curl fallback returned invalid payload.")
        return payload


def fetch_fred_series(
    series_id: str,
    *,
    start: str = PAPER_START,
    end: str = PAPER_END,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    extras: dict[str, str] | None = None,
) -> pd.Series:
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_path = _fred_cache_path(cache_dir, series_id, start, end, extras)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        params = {"id": series_id, "cosd": start, "coed": end}
        if extras:
            params.update(extras)
        url = f"{FRED_BASE_URL}?{urlencode(params)}"
        payload = _fetch_fred_payload(url)
        cache_path.write_bytes(payload)

    frame = pd.read_csv(io.BytesIO(payload))
    if frame.empty:
        raise RuntimeError(f"FRED series {series_id} returned no data.")

    date_col = "DATE" if "DATE" in frame.columns else frame.columns[0]
    value_col = series_id if series_id in frame.columns else frame.columns[-1]
    frame = frame.rename(columns={date_col: "date", value_col: "value"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    series = frame.set_index("date")["value"].sort_index()
    series.name = series_id
    return series


def fetch_yahoo_history(
    symbol: str,
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    interval: str = "1d",
) -> pd.DataFrame:
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_path = _yahoo_cache_path(cache_dir, symbol, start, end, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["Date"])
        return frame.set_index("Date").sort_index()

    frame = yf.download(
        symbol,
        start=start,
        end=(pd.Timestamp(end) + pd.offsets.Day(1)).strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError(f"Yahoo download returned no rows for {symbol}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]

    frame = frame.reset_index()
    frame.to_csv(cache_path, index=False)
    return frame.set_index("Date").sort_index()


def _to_month_end(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = pd.to_datetime(result.index).to_period("M").to_timestamp("M")
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def _resample_month_end(series: pd.Series, how: str) -> pd.Series:
    if how == "last":
        result = series.resample("ME").last()
    elif how == "mean":
        result = series.resample("ME").mean()
    else:
        raise ValueError(f"Unsupported monthly aggregation: {how}")
    return result.sort_index()


def _annualized_monthly_realized_vol(price_frame: pd.DataFrame) -> pd.Series:
    close = price_frame["Close"].copy()
    returns = close.pct_change().dropna()
    realized = returns.groupby(returns.index.to_period("M")).std() * np.sqrt(252.0) * 100.0
    realized.index = realized.index.to_timestamp("M")
    realized.name = "realized_vol"
    return realized.sort_index()


def _combine_growth_raw(cache_dir: Path) -> pd.DataFrame:
    gs10 = _to_month_end(fetch_fred_series("GS10", cache_dir=cache_dir))
    fedfunds = _to_month_end(fetch_fred_series("FEDFUNDS", cache_dir=cache_dir))
    baa = _to_month_end(fetch_fred_series("BAA", cache_dir=cache_dir))
    permits = _to_month_end(fetch_fred_series("PERMIT", cache_dir=cache_dir))
    claims = _resample_month_end(fetch_fred_series("IC4WSA", cache_dir=cache_dir), "mean")

    vxo = _resample_month_end(fetch_fred_series("VXOCLS", start="1986-01-01", end=PAPER_END, cache_dir=cache_dir), "mean")
    spx_daily = fetch_yahoo_history("^GSPC", start="1975-01-01", end=PAPER_END, cache_dir=cache_dir)
    realized_vol = _annualized_monthly_realized_vol(spx_daily)

    volatility = realized_vol.copy()
    volatility.loc[vxo.index.min() :] = vxo.reindex(volatility.loc[vxo.index.min() :].index).combine_first(volatility.loc[vxo.index.min() :])
    volatility = volatility.combine_first(vxo)
    volatility.name = "volatility_proxy"

    frame = pd.concat(
        {
            "yield_spread": gs10 - fedfunds,
            "credit_spread": baa - gs10,
            "jobless_claims": claims,
            "building_permits": permits,
            "volatility_proxy": volatility,
        },
        axis=1,
    ).sort_index()
    return frame


def _combine_asset_returns(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    stocks = _to_month_end(fetch_fred_series("SPASTT01USM657N", cache_dir=cache_dir)) / 100.0
    stocks.name = "Stocks"

    bonds_index = _to_month_end(fetch_fred_series("BAMLCC0A0CMTRIV", cache_dir=cache_dir))
    bonds = bonds_index.ffill().pct_change(fill_method=None)
    bonds.name = "Bonds"

    commodities_index = _to_month_end(fetch_fred_series("PPIACO", cache_dir=cache_dir))
    commodities = commodities_index.pct_change(fill_method=None)
    commodities.name = "Commodities"

    risk_free = _to_month_end(fetch_fred_series("TB3MS", cache_dir=cache_dir)) / 100.0 / 12.0
    risk_free.name = "RiskFree"

    asset_returns = pd.concat([stocks, bonds, commodities], axis=1).sort_index()
    return asset_returns, risk_free


def load_raw_model_data(cache_dir: Path = DEFAULT_CACHE_DIR) -> RawModelData:
    cache_dir = _ensure_cache_dir(cache_dir)
    growth_raw = _combine_growth_raw(cache_dir)

    cpi = _to_month_end(fetch_fred_series("CPIAUCSL", cache_dir=cache_dir))
    cpi.name = "CPIAUCSL"
    uig = _to_month_end(fetch_fred_series("UIGFULL", start="1995-01-01", end=PAPER_END, cache_dir=cache_dir))
    uig.name = "UIGFULL"

    asset_returns, risk_free = _combine_asset_returns(cache_dir)
    vix_reference = _resample_month_end(fetch_fred_series("VXOCLS", start="1986-01-01", end=PAPER_END, cache_dir=cache_dir), "mean")
    vix_reference.name = "VXOCLS"

    notes = [
        "FRED clean CSV endpoint with curl fallback for GS10, FEDFUNDS, BAA, IC4WSA, PERMIT, CPIAUCSL, UIGFULL, VXOCLS, SPASTT01USM657N, BAMLCC0A0CMTRIV, PPIACO, and TB3MS.",
        "Yahoo Finance direct download for ^GSPC daily closes to compute pre-1986 realized volatility.",
        f"Growth raw window: {growth_raw.dropna(how='all').index.min():%Y-%m} to {growth_raw.dropna(how='all').index.max():%Y-%m}.",
        f"Asset return window: {asset_returns.dropna(how='all').index.min():%Y-%m} to {asset_returns.dropna(how='all').index.max():%Y-%m}.",
    ]

    return RawModelData(
        growth_raw=growth_raw,
        cpi=cpi,
        uig=uig,
        asset_returns=asset_returns,
        risk_free=risk_free,
        vix_reference=vix_reference,
        notes=notes,
    )


def describe_series_ranges(series_map: Iterable[tuple[str, pd.Series]]) -> list[str]:
    lines: list[str] = []
    for label, series in series_map:
        non_null = series.dropna()
        if non_null.empty:
            continue
        lines.append(f"{label}: {non_null.index.min():%Y-%m} to {non_null.index.max():%Y-%m} ({len(non_null)} obs)")
    return lines
