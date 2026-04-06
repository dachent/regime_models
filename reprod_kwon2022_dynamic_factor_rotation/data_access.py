from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import shutil


USER_AGENT = "regime-models-kwon2022/1.0"
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "cache"
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FAMAFRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
MISSING_SENTINELS = {"-99.99", "-999", "-999.0", "-999.00"}


@dataclass(frozen=True)
class KwonRawData:
    gs10: pd.Series
    fedfunds: pd.Series
    baa: pd.Series
    ic4wsa: pd.Series
    permit: pd.Series
    vixcls: pd.Series
    sp500_close: pd.Series
    factors: pd.DataFrame
    fred_vintage_date: str | None
    data_end: str


def today_iso() -> str:
    return date.today().isoformat()


def next_day_iso(value: str) -> str:
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


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
            delay = backoff_seconds * (attempt + 1)
            if delay > 0:
                import time

                time.sleep(delay)
    raise RuntimeError(str(last_error) if last_error is not None else "Unknown fetch error.")


def download_bytes_via_node(url: str) -> bytes:
    if shutil.which("node") is None:
        raise RuntimeError("node is not installed.")
    script = """
const url = process.argv[1];
const headers = { "user-agent": process.argv[2] };
(async () => {
  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  process.stdout.write(buffer);
})().catch((error) => {
  console.error(String(error && error.message ? error.message : error));
  process.exit(1);
});
""".strip()
    completed = subprocess.run(
        ["node", "-e", script, url, USER_AGENT],
        check=True,
        capture_output=True,
        text=False,
        timeout=180,
    )
    return completed.stdout


def download_bytes(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT}
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
            return download_bytes_via_node(url)
        except Exception:  # noqa: BLE001
            pass

        curl_args = [
            "curl.exe",
            "-L",
            "--silent",
            "--show-error",
            "--compressed",
            "-A",
            USER_AGENT,
            url,
        ]
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


def build_fred_url(
    series_id: str,
    start: str,
    end: str,
    *,
    vintage_date: str | None = None,
) -> str:
    params = {"id": series_id, "cosd": start, "coed": end}
    if vintage_date:
        params["vintage_date"] = vintage_date
    return f"{FRED_BASE_URL}?{urllib.parse.urlencode(params)}"


def load_fred_series(
    series_id: str,
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    vintage_date: str | None = None,
) -> pd.Series:
    url = build_fred_url(series_id, start, end, vintage_date=vintage_date)
    cache_file = cache_path(
        cache_dir,
        "fred",
        f"{series_id}_{stable_hash({'url': url, 'vintage': vintage_date})}",
        ".csv",
    )
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(url))
    frame = pd.read_csv(io.BytesIO(payload))
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise RuntimeError(f"Unexpected FRED schema for {series_id}.")
    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    series = pd.Series(values.values, index=dates, name=series_id).dropna().sort_index()
    if series.empty:
        raise RuntimeError(f"FRED returned no usable data for {series_id}.")
    return series


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for the Yahoo-compatible data path.") from exc
    return yf


def normalize_yahoo_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            frame = frame[ticker]
        elif ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame = frame.droplevel(-1, axis=1)

    data = frame.reset_index().copy()
    date_column = str(data.columns[0])
    data = data.rename(columns={date_column: "Date"})
    if "Close" not in data.columns:
        raise RuntimeError(f"Yahoo schema for {ticker} did not include Close.")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if hasattr(data["Date"].dt, "tz") and data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_localize(None)
    return data


def fetch_yahoo_close(
    ticker: str,
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    cache_file = cache_path(
        cache_dir,
        "yahoo",
        f"{ticker.replace('^', '_')}_{stable_hash({'start': start, 'end': end})}",
        ".csv",
    )
    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["Date"])
        return pd.Series(pd.to_numeric(cached["Close"], errors="coerce").values, index=cached["Date"], name=ticker).dropna()

    yf = import_yfinance()

    def fetch_frame() -> pd.DataFrame:
        raw = yf.download(
            tickers=ticker,
            start=start,
            end=next_day_iso(end),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=60,
        )
        return normalize_yahoo_frame(raw, ticker)

    frame = fetch_with_retries(fetch_frame, attempts=2, backoff_seconds=2.0)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    output = pd.DataFrame({"Date": pd.to_datetime(frame["Date"]), "Close": close}).dropna()
    if output.empty:
        raise RuntimeError(f"Yahoo returned no usable close data for {ticker}.")
    output.to_csv(cache_file, index=False)
    return pd.Series(output["Close"].values, index=output["Date"], name=ticker).sort_index()


def famafrench_feed_url(feed_id: str) -> str:
    if not feed_id or not feed_id.upper().endswith("_CSV"):
        raise ValueError("Fama/French feed id must end in _CSV.")
    return f"{FAMAFRENCH_BASE_URL}/{urllib.parse.quote(feed_id)}.zip"


def load_famafrench_csv_text(feed_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    url = famafrench_feed_url(feed_id)
    cache_file = cache_path(cache_dir, "famafrench", f"{feed_id}_{stable_hash(url)}", ".zip")
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(url))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"Fama/French zip for {feed_id} did not contain a CSV.")
        return archive.read(members[0]).decode("utf-8-sig", errors="replace")


def parse_famafrench_monthly_section(text: str, ncols: int) -> list[list[float | str | None]]:
    rows: list[list[float | str | None]] = []
    reader = csv.reader(text.splitlines())
    for parts in reader:
        if len(parts) < ncols:
            continue
        token = parts[0].strip()
        if not (token.isdigit() and len(token) == 6):
            continue
        parsed: list[float | str | None] = [token]
        for value in parts[1:ncols]:
            cleaned = value.strip()
            if cleaned in MISSING_SENTINELS or cleaned == "":
                parsed.append(None)
            else:
                parsed.append(float(cleaned))
        rows.append(parsed)
    if not rows:
        raise RuntimeError("No monthly rows were found in the Fama/French feed.")
    return rows


def month_start_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.to_period("M").to_timestamp()


def load_factor_data(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    start: str = "1967-01-01",
    end: str = "2021-10-31",
) -> pd.DataFrame:
    ff5_rows = parse_famafrench_monthly_section(
        load_famafrench_csv_text("F-F_Research_Data_5_Factors_2x3_CSV", cache_dir=cache_dir),
        7,
    )
    ff5 = pd.DataFrame(ff5_rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    ff5["Date"] = pd.to_datetime(ff5["Date"].astype(str), format="%Y%m")
    ff5 = ff5.set_index("Date") / 100.0

    momentum_rows = parse_famafrench_monthly_section(
        load_famafrench_csv_text("F-F_Momentum_Factor_CSV", cache_dir=cache_dir),
        2,
    )
    momentum = pd.DataFrame(momentum_rows, columns=["Date", "Mom"])
    momentum["Date"] = pd.to_datetime(momentum["Date"].astype(str), format="%Y%m")
    momentum = momentum.set_index("Date") / 100.0

    factors = ff5[["SMB", "HML", "RMW", "CMA"]].join(momentum["Mom"], how="inner")
    factors.columns = ["Size", "Value", "Profitability", "Investment", "Momentum"]
    factors = factors[["Size", "Value", "Momentum", "Profitability", "Investment"]]
    factors.index = month_start_index(pd.DatetimeIndex(factors.index))
    return factors.loc[(factors.index >= pd.Timestamp(start)) & (factors.index <= pd.Timestamp(end))].dropna()


def load_kwon_raw_data(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    data_end: str = "2021-10-31",
    fred_vintage_date: str | None = "2021-10-31",
) -> KwonRawData:
    common_start = "1965-01-01"
    gs10 = load_fred_series("GS10", start=common_start, end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    fedfunds = load_fred_series("FEDFUNDS", start=common_start, end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    baa = load_fred_series("BAA", start=common_start, end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    ic4wsa = load_fred_series("IC4WSA", start=common_start, end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    permit = load_fred_series("PERMIT", start=common_start, end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    vixcls = load_fred_series("VIXCLS", start="1990-01-01", end=data_end, cache_dir=cache_dir, vintage_date=fred_vintage_date)
    sp500_close = fetch_yahoo_close("^GSPC", start=common_start, end=data_end, cache_dir=cache_dir)
    factors = load_factor_data(cache_dir=cache_dir, start="1967-01-01", end=data_end)

    return KwonRawData(
        gs10=gs10,
        fedfunds=fedfunds,
        baa=baa,
        ic4wsa=ic4wsa,
        permit=permit,
        vixcls=vixcls,
        sp500_close=sp500_close,
        factors=factors,
        fred_vintage_date=fred_vintage_date,
        data_end=data_end,
    )
