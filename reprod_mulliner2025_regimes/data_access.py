from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from model_core import compute_bond_total_returns, compute_sb_correlation


USER_AGENT = "regime-models-amdt-compatible/1.0"
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "cache"
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MACROTRENDS_PAGE_ID_PATTERNS = [
    re.compile(r"generateChart\('(?P<page_id>\d+)'"),
    re.compile(r'var pageId = "(?P<page_id>\d+)"'),
]
COPPER_PAGE_URL = "https://www.macrotrends.net/datasets/1476/copper-prices-historical-chart-data"
FAMAFRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
MISSING_SENTINELS = {"-99.99", "-999", "-999.0", "-999.00"}
HISTORY_COLUMNS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]
PRODUCTION_YAHOO_TICKERS = ["^GSPC", "^TNX", "^VIX"]
SCENARIO_YAHOO_TICKERS = ["^GSPC", "^SP500TR", "^TNX", "^IRX", "^VIX", "AGG", "^FVX"]


def today_iso() -> str:
    return date.today().isoformat()


def parse_date_token(value: str) -> date:
    token = value.strip().lower()
    today = date.today()
    if token == "today":
        return today
    if token == "yesterday":
        return today - timedelta(days=1)
    return date.fromisoformat(value)


def cache_path(cache_dir: Path, provider: str, stem: str, suffix: str) -> Path:
    path = cache_dir / provider / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


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

        curl_args = [
            "curl.exe",
            "-L",
            "--silent",
            "--show-error",
            "--compressed",
            "-A",
            USER_AGENT,
        ]
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


def fetch_with_retries(fetcher: Callable[[], object], attempts: int = 2, backoff_seconds: float = 2.0) -> object:
    last_error: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return fetcher()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            sleep_seconds = backoff_seconds * (attempt + 1)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raise RuntimeError(str(last_error) if last_error is not None else "Unknown fetch error.")


def build_fred_url(
    series_id: str,
    start: str,
    end: str,
    *,
    transformation: str | None = None,
    fq: str | None = None,
    fam: str | None = None,
    vintage_date: str | None = None,
) -> str:
    params = {
        "id": series_id,
        "cosd": start,
        "coed": end,
    }
    if transformation and transformation != "lin":
        params["transformation"] = transformation
    if fq:
        params["fq"] = fq
        params["fam"] = fam or "avg"
    elif fam:
        raise ValueError("FRED aggregation requires frequency override.")
    if vintage_date:
        params["vintage_date"] = vintage_date
    return f"{FRED_BASE_URL}?{urllib.parse.urlencode(params)}"


def load_fred_series(
    series_id: str,
    *,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    transformation: str | None = None,
    fq: str | None = None,
    fam: str | None = None,
    vintage_date: str | None = None,
    month_end_index: bool = False,
) -> pd.Series:
    url = build_fred_url(
        series_id,
        start,
        end,
        transformation=transformation,
        fq=fq,
        fam=fam,
        vintage_date=vintage_date,
    )
    cache_file = cache_path(cache_dir, "fred", f"{series_id}_{stable_hash({'url': url})}", ".csv")
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(url))
    if payload.lstrip().startswith(b"<"):
        raise RuntimeError(f"FRED returned HTML instead of CSV for {series_id}.")

    frame = pd.read_csv(io.BytesIO(payload))
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise RuntimeError(f"Unexpected FRED schema for {series_id}.")

    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    if month_end_index:
        dates = dates + pd.offsets.MonthEnd(0)
    values = pd.to_numeric(frame[series_id], errors="coerce")
    series = pd.Series(values.values, index=dates, name=series_id)
    return series.dropna().sort_index()


def extract_macrotrends_page_id(page_url: str, html: str) -> str:
    for pattern in MACROTRENDS_PAGE_ID_PATTERNS:
        match = pattern.search(html)
        if match:
            return match.group("page_id")

    parsed = urllib.parse.urlparse(page_url)
    path_match = re.search(r"/datasets/(?P<page_id>\d+)/", parsed.path)
    if path_match:
        return path_match.group("page_id")

    raise RuntimeError("Could not extract Macrotrends page_id from the page HTML or URL.")


def fetch_macrotrends_payload(page_url: str, *, chart_frequency: str = "D", cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, object]:
    page_cache = cache_path(cache_dir, "macrotrends", f"page_{stable_hash(page_url)}", ".html")
    page_html = load_or_download_bytes(page_cache, lambda: download_bytes(page_url)).decode("utf-8", errors="replace")
    page_id = extract_macrotrends_page_id(page_url, page_html)
    endpoint = f"https://www.macrotrends.net/economic-data/{page_id}/{chart_frequency}"
    payload_cache = cache_path(
        cache_dir,
        "macrotrends",
        f"payload_{page_id}_{chart_frequency}_{stable_hash(endpoint)}",
        ".json",
    )
    payload_bytes = load_or_download_bytes(payload_cache, lambda: download_bytes(endpoint, referer=page_url))
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload or "metadata" not in payload:
        raise RuntimeError("Macrotrends returned an unexpected payload.")
    return {
        "page_url": page_url,
        "page_id": page_id,
        "chart_frequency": chart_frequency,
        "endpoint": endpoint,
        "metadata": payload.get("metadata"),
        "data": payload.get("data"),
    }


def load_macrotrends_series(page_url: str, *, chart_frequency: str = "D", cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    payload = fetch_macrotrends_payload(page_url, chart_frequency=chart_frequency, cache_dir=cache_dir)
    rows = payload["data"]
    if not isinstance(rows, list):
        raise RuntimeError("Macrotrends payload did not contain a row list.")
    parsed_rows = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        parsed_rows.append(row[:2])
    frame = pd.DataFrame(parsed_rows, columns=["EpochMs", "Value"])
    frame["Date"] = pd.to_datetime(frame["EpochMs"], unit="ms", errors="coerce", utc=True).dt.tz_localize(None)
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Value"]).sort_values("Date")
    return pd.Series(frame["Value"].values, index=frame["Date"], name=str(payload.get("page_id")))


def famafrench_feed_url(feed_id: str) -> str:
    cleaned = feed_id.strip()
    if not cleaned:
        raise ValueError("Fama/French feed id must not be empty.")
    if not cleaned.upper().endswith("_CSV"):
        raise ValueError("Fama/French feed id must be the exact CSV zip stem ending in _CSV.")
    return f"{FAMAFRENCH_BASE_URL}/{urllib.parse.quote(cleaned)}.zip"


def load_famafrench_csv_text(feed_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    url = famafrench_feed_url(feed_id)
    cache_file = cache_path(cache_dir, "famafrench", f"{feed_id}_{stable_hash(url)}", ".zip")
    payload = load_or_download_bytes(cache_file, lambda: download_bytes(url))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_members = [member for member in archive.namelist() if member.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("Fama/French zip did not contain a CSV member.")
        return archive.read(csv_members[0]).decode("utf-8-sig", errors="replace")


def parse_famafrench_monthly_section(text: str, ncols: int) -> list[list[float | int | str | None]]:
    rows: list[list[float | int | str | None]] = []
    reader = csv.reader(text.splitlines())
    for parts in reader:
        if len(parts) < ncols:
            continue
        token = parts[0].strip()
        if not (token.isdigit() and len(token) == 6):
            continue
        parsed: list[float | int | str | None] = [token]
        for value in parts[1:ncols]:
            cleaned = value.strip()
            if cleaned in MISSING_SENTINELS or cleaned == "":
                parsed.append(None)
                continue
            parsed.append(float(cleaned))
        rows.append(parsed)
    if not rows:
        raise RuntimeError("No monthly rows were found in the Fama/French feed.")
    return rows


def load_fama_french_factors(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
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

    factors = ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(momentum["Mom"], how="inner")
    factors.columns = ["Market", "Size", "Value", "Profitability", "Investment", "Momentum"]
    return factors.dropna().sort_index()


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for the Yahoo-compatible data path.") from exc
    return yf


def normalize_single_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = frame.reset_index().copy()
    first_column = str(data.columns[0])
    data = data.rename(columns={first_column: "Date"})
    for column in HISTORY_COLUMNS[2:]:
        if column not in data.columns:
            data[column] = pd.NA
    data.insert(0, "Ticker", ticker)
    data = data.dropna(
        axis=0,
        how="all",
        subset=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"],
    )
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if hasattr(data["Date"].dt, "tz") and data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_localize(None)
    return data[HISTORY_COLUMNS]


def normalize_download_result(raw: pd.DataFrame | None, requested_tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if any(ticker in level0 for ticker in requested_tickers):
            for ticker in requested_tickers:
                if ticker in level0:
                    frames.append(normalize_single_ticker_frame(raw[ticker].copy(), ticker))
        else:
            level1 = set(raw.columns.get_level_values(1))
            for ticker in requested_tickers:
                if ticker in level1:
                    frames.append(normalize_single_ticker_frame(raw.xs(ticker, axis=1, level=1).copy(), ticker))
    else:
        frames.append(normalize_single_ticker_frame(raw.copy(), requested_tickers[0]))

    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"], kind="stable").reset_index(drop=True)
    return combined[HISTORY_COLUMNS]


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def fetch_yahoo_history_frame(
    tickers: list[str],
    *,
    start: str,
    end: str = "today",
    interval: str = "1d",
    prepost: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 10,
    threads: int = 4,
    timeout_seconds: int = 60,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    yf = import_yfinance()
    cache_key = stable_hash(
        {
            "tickers": tickers,
            "start": start,
            "end": end,
            "interval": interval,
            "prepost": prepost,
            "batch_size": batch_size,
        }
    )
    cache_file = cache_path(cache_dir, "yahoo", f"history_{cache_key}", ".csv")
    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["Date"])
        for column in HISTORY_COLUMNS:
            if column not in cached.columns:
                cached[column] = pd.NA
        return cached[HISTORY_COLUMNS]

    requested = []
    seen: set[str] = set()
    for ticker in tickers:
        cleaned = str(ticker).strip()
        if cleaned and cleaned not in seen:
            requested.append(cleaned)
            seen.add(cleaned)
    if not requested:
        raise ValueError("Provide at least one ticker for Yahoo history.")

    start_date = parse_date_token(start).isoformat()
    end_date = parse_date_token(end).isoformat()

    frames: list[pd.DataFrame] = []
    for batch in batched(requested, batch_size):
        def fetch_batch() -> pd.DataFrame:
            raw = yf.download(
                tickers=batch if len(batch) > 1 else batch[0],
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=prepost,
                actions=True,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=threads,
                timeout=timeout_seconds,
            )
            return normalize_download_result(raw, batch)

        batch_frame = fetch_with_retries(fetch_batch, attempts=retry_attempts, backoff_seconds=retry_backoff_seconds)
        frames.append(batch_frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HISTORY_COLUMNS)
    combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"], kind="stable").reset_index(drop=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_file, index=False)
    return combined[HISTORY_COLUMNS]


def yahoo_close_map(
    tickers: list[str],
    *,
    start: str,
    end: str = "today",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, pd.Series]:
    frame = fetch_yahoo_history_frame(tickers, start=start, end=end, cache_dir=cache_dir)
    output: dict[str, pd.Series] = {}
    for ticker, ticker_frame in frame.groupby("Ticker", sort=False):
        close_values = pd.to_numeric(ticker_frame["Close"], errors="coerce")
        series = pd.Series(close_values.values, index=pd.to_datetime(ticker_frame["Date"]), name=ticker)
        output[ticker] = series.dropna().sort_index()
    return output


def load_eia_daily_wti_series(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    page_url = "https://www.eia.gov/dnav/pet/hist/RWTCD.htm"
    page_cache = cache_path(cache_dir, "eia", f"page_{stable_hash(page_url)}", ".html")
    page_html = load_or_download_bytes(page_cache, lambda: download_bytes(page_url)).decode("utf-8", errors="replace")
    match = re.search(r'href=["\'](?P<href>\.\./hist_xls/RWTCd\.xls)["\']', page_html, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Could not locate the linked EIA history workbook for RWTCd.")

    workbook_url = urllib.parse.urljoin(page_url, match.group("href"))
    workbook_cache = cache_path(cache_dir, "eia", f"rwtcd_{stable_hash(workbook_url)}", ".xls")
    workbook_bytes = load_or_download_bytes(workbook_cache, lambda: download_bytes(workbook_url))

    try:
        import xlrd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xlrd is required for the EIA history workbook path.") from exc

    workbook = xlrd.open_workbook(file_contents=workbook_bytes)
    sheet = workbook.sheet_by_name("Data 1")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row_index in range(3, sheet.nrows):
        date_cell = sheet.cell_value(row_index, 0)
        value_cell = sheet.cell_value(row_index, 1)
        if not isinstance(date_cell, float) or not isinstance(value_cell, float) or value_cell <= 0:
            continue
        parsed_date = datetime(1899, 12, 30) + timedelta(days=int(date_cell))
        dates.append(pd.Timestamp(parsed_date))
        values.append(float(value_cell))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="RWTCd").sort_index()


def monthly_mean(series: pd.Series) -> pd.Series:
    return series.resample("ME").mean().dropna()


def monthly_last(series: pd.Series) -> pd.Series:
    return series.resample("ME").last().dropna()


def load_production_data(cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[dict[str, pd.Series], pd.DataFrame, list[str]]:
    notes: list[str] = []
    yahoo = yahoo_close_map(PRODUCTION_YAHOO_TICKERS, start="1920-01-01", cache_dir=cache_dir)
    gspc_close = yahoo["^GSPC"]
    gspc_monthly = monthly_last(gspc_close)
    gspc_daily_returns = gspc_close.pct_change().dropna()
    notes.append(describe_series("S&P 500", gspc_monthly))

    tnx_daily = yahoo["^TNX"].dropna()
    notes.append(describe_series("10Y yield", tnx_daily))

    vix_monthly = monthly_last(yahoo["^VIX"])
    notes.append(describe_series("VIX", vix_monthly))

    gs10 = load_fred_series("GS10", start="1953-04-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    tb3ms = load_fred_series("TB3MS", start="1934-01-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    oil = load_fred_series("WTISPLC", start="1946-01-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    notes.append(describe_series("GS10", gs10))
    notes.append(describe_series("TB3MS", tb3ms))
    notes.append(describe_series("WTISPLC", oil))

    copper_daily = load_macrotrends_series(COPPER_PAGE_URL, chart_frequency="D", cache_dir=cache_dir)
    copper_daily = copper_daily[copper_daily > 0]
    copper_monthly = monthly_last(copper_daily)
    notes.append(describe_series("Copper", copper_monthly))

    realized_vol = monthly_last(gspc_daily_returns.rolling(63).std() * (252.0 ** 0.5) * 100.0)
    volatility = pd.concat([realized_vol[realized_vol.index < vix_monthly.index[0]], vix_monthly]).sort_index()
    volatility = volatility[~volatility.index.duplicated(keep="last")]

    bond_returns = compute_bond_total_returns(tnx_daily)
    stock_bond = compute_sb_correlation(gspc_daily_returns, bond_returns)

    econ_variables = {
        "Market": gspc_monthly,
        "Yield_curve": (gs10 - tb3ms).dropna(),
        "Oil": oil,
        "Copper": copper_monthly,
        "Monetary_policy": tb3ms,
        "Volatility": volatility,
        "Stock_bond": stock_bond,
    }
    factors = load_fama_french_factors(cache_dir=cache_dir)
    notes.append(describe_series("Factors", pd.Series(index=factors.index, data=1.0)))
    return econ_variables, factors, notes


def load_all_scenario_sources(cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[dict[str, pd.Series], list[str]]:
    notes: list[str] = []
    sources: dict[str, pd.Series] = {}

    yahoo = yahoo_close_map(SCENARIO_YAHOO_TICKERS, start="1920-01-01", cache_dir=cache_dir)
    gspc_close = yahoo["^GSPC"]
    sources["gspc_m"] = monthly_last(gspc_close)
    sources["gspc_d"] = gspc_close.pct_change().dropna()
    notes.append(describe_series("^GSPC", sources["gspc_m"]))

    if "^SP500TR" in yahoo:
        sources["sp500tr_m"] = monthly_last(yahoo["^SP500TR"])
        splice_date = sources["sp500tr_m"].index[0]
        sources["synth_tr_m"] = pd.concat(
            [sources["gspc_m"][sources["gspc_m"].index < splice_date], sources["sp500tr_m"]]
        ).sort_index()
        sources["synth_tr_m"] = sources["synth_tr_m"][~sources["synth_tr_m"].index.duplicated(keep="last")]
        notes.append(describe_series("^SP500TR", sources["sp500tr_m"]))

    sources["tnx_d"] = yahoo["^TNX"].dropna()
    sources["tnx_avg"] = monthly_mean(sources["tnx_d"])
    sources["tnx_eom"] = monthly_last(sources["tnx_d"])
    notes.append(describe_series("^TNX", sources["tnx_d"]))
    notes.append(describe_series("^TNX avg", sources["tnx_avg"]))

    sources["irx_d"] = yahoo["^IRX"].dropna()
    sources["irx_avg"] = monthly_mean(sources["irx_d"])
    sources["irx_eom"] = monthly_last(sources["irx_d"])
    notes.append(describe_series("^IRX", sources["irx_d"]))
    notes.append(describe_series("^IRX avg", sources["irx_avg"]))

    sources["vix_m"] = monthly_last(yahoo["^VIX"])
    notes.append(describe_series("^VIX", sources["vix_m"]))

    if "AGG" in yahoo:
        sources["agg_d"] = yahoo["AGG"].pct_change().dropna()
        notes.append(describe_series("AGG", yahoo["AGG"]))

    if "^FVX" in yahoo:
        sources["fvx_d"] = yahoo["^FVX"].dropna()
        notes.append(describe_series("^FVX", sources["fvx_d"]))

    sources["gs10_avg"] = load_fred_series("GS10", start="1953-04-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    sources["tb3ms_avg"] = load_fred_series("TB3MS", start="1934-01-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    sources["oil_fred"] = load_fred_series("WTISPLC", start="1946-01-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
    notes.append(describe_series("GS10 avg", sources["gs10_avg"]))
    notes.append(describe_series("TB3MS avg", sources["tb3ms_avg"]))
    notes.append(describe_series("Oil FRED", sources["oil_fred"]))

    copper_daily = load_macrotrends_series(COPPER_PAGE_URL, chart_frequency="D", cache_dir=cache_dir)
    copper_daily = copper_daily[copper_daily > 0]
    sources["cu_eom"] = monthly_last(copper_daily)
    notes.append(describe_series("Copper EOM", sources["cu_eom"]))

    try:
        sources["oil_eia_d"] = load_eia_daily_wti_series(cache_dir=cache_dir)
        sources["oil_eia_eom"] = monthly_last(sources["oil_eia_d"])
        sources["oil_splice"] = pd.concat(
            [
                sources["oil_fred"][sources["oil_fred"].index < sources["oil_eia_eom"].index[0]],
                sources["oil_eia_eom"],
            ]
        ).sort_index()
        sources["oil_splice"] = sources["oil_splice"][~sources["oil_splice"].index.duplicated(keep="last")]
        notes.append(describe_series("Oil EIA", sources["oil_eia_eom"]))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Oil EIA optional path unavailable: {exc}")

    try:
        sources["cu_fred"] = load_fred_series("PCOPPUSDM", start="1992-01-01", end=today_iso(), cache_dir=cache_dir, month_end_index=True)
        notes.append(describe_series("Copper FRED", sources["cu_fred"]))
        sources["cu_splice"] = pd.concat(
            [
                sources["cu_eom"][sources["cu_eom"].index < sources["cu_fred"].index[0]],
                sources["cu_fred"],
            ]
        ).sort_index()
        sources["cu_splice"] = sources["cu_splice"][~sources["cu_splice"].index.duplicated(keep="last")]
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Copper FRED optional path unavailable: {exc}")

    for label, window_days in [("21d", 21), ("63d", 63)]:
        realized = monthly_last(sources["gspc_d"].rolling(window_days).std() * (252.0 ** 0.5) * 100.0)
        pre_vix = realized[realized.index < sources["vix_m"].index[0]]
        spliced = pd.concat([pre_vix, sources["vix_m"]]).sort_index()
        sources[f"vol_{label}"] = spliced[~spliced.index.duplicated(keep="last")]

    bond_returns_10y = compute_bond_total_returns(sources["tnx_d"])
    yield_changes_10y = sources["tnx_d"].diff().dropna()
    sources["sb_bondtr"] = compute_sb_correlation(sources["gspc_d"], bond_returns_10y)
    sources["sb_yldchg"] = compute_sb_correlation(sources["gspc_d"], yield_changes_10y)
    sources["sb_inverted"] = -sources["sb_bondtr"]
    sources["sb_5yr"] = compute_sb_correlation(sources["gspc_d"], bond_returns_10y, window=1260, min_periods=800)

    if "agg_d" in sources:
        agg_splice = pd.concat(
            [bond_returns_10y[bond_returns_10y.index < sources["agg_d"].index[0]], sources["agg_d"]]
        ).sort_index()
        sources["sb_agg"] = compute_sb_correlation(sources["gspc_d"], agg_splice)

    if "fvx_d" in sources:
        bond_returns_5y = compute_bond_total_returns(sources["fvx_d"], maturity=5)
        sources["sb_5ybond"] = compute_sb_correlation(sources["gspc_d"], bond_returns_5y)

    return sources, notes


def describe_series(name: str, series: pd.Series) -> str:
    return f"{name:<16s} {series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m} ({len(series)} obs)"
