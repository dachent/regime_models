from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


USER_AGENT = "regime-models-amdt-compatible/1.0"
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "cache"
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def today_iso() -> str:
    return date.today().isoformat()


def cache_path(cache_dir: Path, provider: str, stem: str, suffix: str) -> Path:
    path = cache_dir / provider / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def fetch_with_retries(fetcher: Callable[[], object], attempts: int = 2, backoff_seconds: float = 2.0) -> object:
    last_error: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return fetcher()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(str(last_error) if last_error is not None else "Unknown fetch error.")


def download_bytes_via_node(url: str, *, referer: str | None = None) -> bytes:
    if shutil.which("node") is None:
        raise RuntimeError("node is not installed.")

    script = """
const url = process.argv[1];
const referer = process.argv[2] || "";
const headers = { "user-agent": process.argv[3] };
if (referer) headers.referer = referer;
(async () => {
  const response = await fetch(url, { headers });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  process.stdout.write(buffer);
})().catch((error) => {
  console.error(String(error && error.message ? error.message : error));
  process.exit(1);
});
""".strip()
    completed = subprocess.run(
        ["node", "-e", script, url, referer or "", USER_AGENT],
        check=True,
        capture_output=True,
        text=False,
        timeout=180,
    )
    return completed.stdout


def download_bytes(url: str, *, referer: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except Exception as first_error:  # noqa: BLE001
        try:
            import requests  # type: ignore
        except ImportError:
            requests = None  # type: ignore[assignment]

        if requests is not None:
            try:
                response = requests.get(url, headers=headers, timeout=120)
                response.raise_for_status()
                return response.content
            except Exception:  # noqa: BLE001
                pass

        try:
            return download_bytes_via_node(url, referer=referer)
        except Exception:  # noqa: BLE001
            pass

        curl_args = ["curl.exe", "-L", "--silent", "--show-error", "--compressed", "-A", USER_AGENT]
        if referer:
            curl_args.extend(["-e", referer])
        curl_args.append(url)
        try:
            completed = subprocess.run(
                curl_args,
                check=True,
                capture_output=True,
                text=False,
                timeout=180,
            )
            return completed.stdout
        except Exception:  # noqa: BLE001
            raise first_error


def load_or_download_bytes(path: Path, fetcher: Callable[[], bytes]) -> bytes:
    if path.exists():
        return path.read_bytes()
    payload = fetch_with_retries(fetcher, attempts=2, backoff_seconds=2.0)
    path.write_bytes(payload)
    return payload


def build_fred_url(series_id: str, start: str, end: str) -> str:
    params = {"id": series_id, "cosd": start, "coed": end}
    return f"{FRED_BASE_URL}?{urllib.parse.urlencode(params)}"


def load_fred_series(series_id: str, *, start: str, end: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    url = build_fred_url(series_id, start, end)
    cache_file = cache_path(cache_dir, "fred", f"{series_id}_{stable_hash(url)}", ".csv")
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(url))
    if payload.lstrip().startswith(b"<"):
        raise RuntimeError(f"FRED returned HTML instead of CSV for {series_id}.")

    frame = pd.read_csv(io.BytesIO(payload))
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise RuntimeError(f"Unexpected FRED schema for {series_id}.")

    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    series = pd.Series(values.values, index=dates, name=series_id)
    return series.dropna().sort_index()


def load_cboe_vix_series(*, start: str, end: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    cache_file = cache_path(cache_dir, "cboe", f"vix_{start}_{end}", ".csv")
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(CBOE_VIX_URL))
    frame = pd.read_csv(io.BytesIO(payload))
    if "DATE" not in frame.columns or "CLOSE" not in frame.columns:
        raise RuntimeError("Unexpected CBOE VIX schema.")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
    series = pd.Series(pd.to_numeric(frame["CLOSE"], errors="coerce").values, index=frame["DATE"], name="VIX")
    return series.dropna().sort_index().loc[start:end]


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for the Yahoo-compatible data path.") from exc
    return yf


def fetch_yahoo_adjusted_close(
    ticker: str,
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    cache_key = stable_hash({"ticker": ticker, "start": start, "end": end, "adj": True})
    cache_file = cache_path(cache_dir, "yahoo", f"{ticker}_{cache_key}", ".parquet")
    if cache_file.exists():
        return pd.read_parquet(cache_file).squeeze("columns")

    yf = import_yfinance()

    def fetch() -> pd.Series:
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
        )
        if frame.empty:
            return pd.Series(dtype=float, name=ticker)
        close = frame["Close"].squeeze().rename(ticker)
        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    series = fetch_with_retries(fetch, attempts=2, backoff_seconds=2.0)
    series.to_frame().to_parquet(cache_file)
    return series


def stitch_proxy_series(primary: pd.Series, proxy: pd.Series) -> pd.Series:
    primary_clean = primary.dropna().sort_index()
    proxy_clean = proxy.dropna().sort_index()
    if primary_clean.empty or proxy_clean.empty:
        return primary_clean if not primary_clean.empty else proxy_clean

    primary_start = primary_clean.index[0]
    if proxy_clean.index[-1] >= primary_start:
        overlap = proxy_clean.reindex(primary_clean.index[:5]).dropna()
        if overlap.empty:
            overlap = proxy_clean[proxy_clean.index <= primary_start].tail(5)
    else:
        overlap = proxy_clean.tail(5)

    if overlap.empty:
        return primary_clean

    ratio = primary_clean.iloc[0] / overlap.mean()
    proxy_scaled = proxy_clean[proxy_clean.index < primary_start] * ratio
    stitched = pd.concat([proxy_scaled, primary_clean]).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched


def fetch_asset_prices(
    assets: dict[str, str],
    proxy_funds: dict[str, str],
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for asset_name, ticker in assets.items():
        primary = fetch_yahoo_adjusted_close(ticker, start=start, end=end, cache_dir=cache_dir)
        series = primary
        proxy_ticker = proxy_funds.get(asset_name)
        if proxy_ticker:
            proxy = fetch_yahoo_adjusted_close(proxy_ticker, start=start, end=end, cache_dir=cache_dir)
            series = stitch_proxy_series(primary, proxy)
        frames[asset_name] = series

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index().ffill(limit=5)


def build_macro_frame(*, start: str, end: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    macro = pd.DataFrame(
        {
            "GS2": load_fred_series("GS2", start=start, end=end, cache_dir=cache_dir),
            "GS10": load_fred_series("GS10", start=start, end=end, cache_dir=cache_dir),
            "TB3MS": load_fred_series("TB3MS", start=start, end=end, cache_dir=cache_dir),
        }
    )
    try:
        macro["VIX"] = load_cboe_vix_series(start=start, end=end, cache_dir=cache_dir)
    except Exception:
        macro["VIX"] = load_fred_series("VIXCLS", start=start, end=end, cache_dir=cache_dir)

    macro.index = pd.to_datetime(macro.index)
    return macro.sort_index().ffill(limit=5)


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def daily_risk_free(macro: pd.DataFrame) -> pd.Series:
    return (macro["TB3MS"] / 100.0 / 252.0).rename("RF")


def excess_log_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rf_aligned = rf.reindex(returns.index).ffill().fillna(0.0)
    return returns.subtract(rf_aligned, axis=0)


def load_all_data(
    assets: dict[str, str],
    proxy_funds: dict[str, str],
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    prices = fetch_asset_prices(assets, proxy_funds, start=start, end=end, cache_dir=cache_dir)
    macro = build_macro_frame(start=start, end=end, cache_dir=cache_dir)
    returns = prices_to_log_returns(prices)
    rf = daily_risk_free(macro)
    ex_returns = excess_log_returns(returns, rf)

    common_index = returns.dropna(how="all").index
    returns = returns.reindex(common_index)
    ex_returns = ex_returns.reindex(common_index)
    macro = macro.reindex(common_index).ffill(limit=5)
    rf = rf.reindex(common_index).ffill(limit=5).fillna(0.0)
    return prices, returns, ex_returns, macro, rf


def build_data_coverage_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in prices.columns:
        series = prices[column].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Asset": column,
                "Start": series.index[0].date().isoformat(),
                "End": series.index[-1].date().isoformat(),
                "Observations": int(len(series)),
            }
        )
    return pd.DataFrame(rows)


def build_source_mapping_table(assets: dict[str, str], proxy_funds: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    descriptions = {
        "LargeCap": "S&P 500 ETF proxy",
        "MidCap": "S&P 400 ETF proxy",
        "SmallCap": "Russell 2000 ETF proxy",
        "EAFE": "MSCI EAFE ETF proxy",
        "EM": "MSCI Emerging Markets ETF proxy",
        "AggBond": "US aggregate bond ETF proxy",
        "Treasury": "Long Treasury ETF proxy",
        "HighYield": "US high yield ETF proxy",
        "Corporate": "US investment-grade corporate ETF proxy",
        "REIT": "US REIT ETF proxy",
        "Commodity": "Broad commodity ETF proxy",
        "Gold": "Gold ETF proxy",
    }
    for asset_name, ticker in assets.items():
        rows.append(
            {
                "Model Variable": asset_name,
                "Production Source": "Yahoo-compatible direct download",
                "Primary Ticker": ticker,
                "Proxy Ticker": proxy_funds.get(asset_name, ""),
                "Endpoint / Contract": "yfinance-compatible adjusted close history",
                "Notes": descriptions.get(asset_name, ""),
            }
        )

    rows.extend(
        [
            {
                "Model Variable": "GS2",
                "Production Source": "FRED",
                "Primary Ticker": "GS2",
                "Proxy Ticker": "",
                "Endpoint / Contract": "fredgraph.csv clean CSV endpoint",
                "Notes": "2-year Treasury yield macro feature",
            },
            {
                "Model Variable": "GS10",
                "Production Source": "FRED",
                "Primary Ticker": "GS10",
                "Proxy Ticker": "",
                "Endpoint / Contract": "fredgraph.csv clean CSV endpoint",
                "Notes": "10-year Treasury yield macro feature",
            },
            {
                "Model Variable": "TB3MS",
                "Production Source": "FRED",
                "Primary Ticker": "TB3MS",
                "Proxy Ticker": "",
                "Endpoint / Contract": "fredgraph.csv clean CSV endpoint",
                "Notes": "3-month T-bill risk-free proxy",
            },
            {
                "Model Variable": "VIX",
                "Production Source": "CBOE",
                "Primary Ticker": "VIX_History.csv",
                "Proxy Ticker": "VIXCLS",
                "Endpoint / Contract": "CBOE direct CSV with FRED fallback",
                "Notes": "Daily VIX close series",
            },
        ]
    )
    return pd.DataFrame(rows)
