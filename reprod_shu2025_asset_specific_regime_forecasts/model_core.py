from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from jumpmodels.jump import JumpModel
from xgboost import XGBClassifier

from data_access import build_data_coverage_table, build_source_mapping_table


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ASSETS = {
    "LargeCap": "SPY",
    "MidCap": "MDY",
    "SmallCap": "IWM",
    "EAFE": "EFA",
    "EM": "EEM",
    "AggBond": "AGG",
    "Treasury": "TLT",
    "HighYield": "HYG",
    "Corporate": "LQD",
    "REIT": "VNQ",
    "Commodity": "GSG",
    "Gold": "GLD",
}

PROXY_FUNDS = {
    "AggBond": "VBMFX",
    "HighYield": "VWEHX",
    "Commodity": "PCRIX",
    "Gold": "USERX",
    "EM": "VEIEX",
    "REIT": "VGSIX",
}

ASSETS_NO_DD = {"AggBond", "Treasury", "Gold"}

GBDT_SMOOTHING_HL = {
    "LargeCap": 8,
    "MidCap": 8,
    "SmallCap": 8,
    "REIT": 8,
    "AggBond": 8,
    "Treasury": 8,
    "Commodity": 4,
    "Gold": 4,
    "Corporate": 2,
    "EM": 0,
    "EAFE": 0,
    "HighYield": 0,
}

DEFAULT_LAMBDAS = {
    "LargeCap": 4.64,
    "MidCap": 4.64,
    "SmallCap": 4.64,
    "EAFE": 4.64,
    "EM": 2.15,
    "AggBond": 10.0,
    "Treasury": 10.0,
    "HighYield": 4.64,
    "Corporate": 4.64,
    "REIT": 4.64,
    "Commodity": 1.0,
    "Gold": 2.15,
}

PAPER_BENCHMARKS = {
    "EW": {"Sharpe": 0.52, "Ann. Return (%)": 6.1, "Max DD (%)": -47.0},
    "EW+Regime": {"Sharpe": 0.81, "Ann. Return (%)": 8.4, "Max DD (%)": -30.0},
    "MinVar": {"Sharpe": 0.71, "Ann. Return (%)": 5.2, "Max DD (%)": -28.0},
    "MinVar+Regime": {"Sharpe": 0.94, "Ann. Return (%)": 6.1, "Max DD (%)": -19.0},
    "MV": {"Sharpe": 0.62, "Ann. Return (%)": 6.8, "Max DD (%)": -35.0},
    "MV+Regime": {"Sharpe": 0.89, "Ann. Return (%)": 8.2, "Max DD (%)": -25.0},
}

LAMBDA_GRID = np.exp(np.linspace(np.log(0.1), np.log(100.0), 25)).tolist()
DATA_START = "1991-01-01"
DATA_END = "2024-01-01"
OOS_START = "2007-01-01"
OOS_END = "2023-12-31"
TRAIN_LOOKBACK_YEARS = 11
VALIDATION_LOOKBACK_YEARS = 5
REFIT_MONTHS = 6
COV_WINDOW_DAYS = 252
RETURN_ESTIMATION_WINDOW_DAYS = 252 * 3
TC_ONE_WAY = 5e-4
GAMMA_RISK = 3.0
GAMMA_TRADE = 1.0
MAX_WEIGHT = 0.40
RANDOM_SEED = 42
MIN_TRAIN_OBS = 252
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


@dataclass
class AssetRefitResult:
    asset: str
    best_lambda: float
    validation_sharpe: float
    probabilities: pd.Series
    train_states: pd.Series


@dataclass
class ModelRun:
    strategy_returns: dict[str, pd.Series]
    stats: pd.DataFrame
    validation_summary: pd.DataFrame
    turnover_summary: pd.DataFrame
    lambda_history: pd.DataFrame
    regime_probabilities: dict[str, pd.Series]
    weights: dict[str, pd.DataFrame]
    data_coverage: pd.DataFrame
    source_mapping: pd.DataFrame


SeedLambdaMap = dict[str, pd.Series]


def generate_refit_dates(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, months: int = REFIT_MONTHS) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    current = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while current <= end_ts:
        dates.append(current)
        current = current + relativedelta(months=months)
    return dates


def ewm_downside_dev(ex_returns: pd.Series, halflife: int) -> pd.Series:
    squared_negative = np.minimum(ex_returns, 0.0) ** 2
    dd = np.sqrt(squared_negative.ewm(halflife=halflife, adjust=True).mean())
    return np.log(dd + 1e-8)


def build_jm_features(ex_returns: pd.Series, asset_name: str) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    if asset_name not in ASSETS_NO_DD:
        for halflife in [5, 21]:
            features[f"dd_log_hl{halflife}"] = ewm_downside_dev(ex_returns, halflife)
    for halflife in [5, 10, 21]:
        avg = ex_returns.ewm(halflife=halflife, adjust=True).mean()
        dd = np.sqrt((np.minimum(ex_returns, 0.0) ** 2).ewm(halflife=halflife, adjust=True).mean())
        features[f"avgret_hl{halflife}"] = avg
        features[f"sortino_hl{halflife}"] = avg / (dd + 1e-8)
    frame = pd.DataFrame(features, index=ex_returns.index)
    frame = (frame - frame.mean()) / (frame.std() + 1e-8)
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_macro_features(macro: pd.DataFrame, large_cap_returns: pd.Series, agg_bond_returns: pd.Series) -> pd.DataFrame:
    gs2 = macro["GS2"].ffill()
    gs10 = macro["GS10"].ffill()
    slope = (gs10 - gs2).ffill()
    vix = macro["VIX"].ffill()
    stock_bond_corr = large_cap_returns.rolling(252, min_periods=63).corr(agg_bond_returns)
    frame = pd.DataFrame(
        {
            "gs2_diff_ewma21": gs2.diff().ewm(halflife=21, adjust=True).mean(),
            "slope_ewma10": slope.ewm(halflife=10, adjust=True).mean(),
            "slope_diff_ewma21": slope.diff().ewm(halflife=21, adjust=True).mean(),
            "vix_logdiff_ewma63": np.log(vix + 1e-8).diff().ewm(halflife=63, adjust=True).mean(),
            "stock_bond_corr_252": stock_bond_corr,
        },
        index=macro.index,
    )
    return frame.replace([np.inf, -np.inf], np.nan).ffill(limit=10)


def build_gbdt_features(ex_returns: pd.Series, macro_features: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for halflife in [5, 21]:
        features[f"dd_log_hl{halflife}"] = ewm_downside_dev(ex_returns, halflife)
    for halflife in [5, 10, 21]:
        avg = ex_returns.ewm(halflife=halflife, adjust=True).mean()
        dd = np.sqrt((np.minimum(ex_returns, 0.0) ** 2).ewm(halflife=halflife, adjust=True).mean())
        features[f"avgret_hl{halflife}"] = avg
        features[f"sortino_hl{halflife}"] = avg / (dd + 1e-8)
    return_features = pd.DataFrame(features, index=ex_returns.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    macro_aligned = macro_features.reindex(ex_returns.index).ffill(limit=10).fillna(0.0)
    return pd.concat([return_features, macro_aligned], axis=1)


def fit_jm_sorted(features: pd.DataFrame, ex_returns: pd.Series, lam: float) -> np.ndarray:
    jm = JumpModel(n_components=2, jump_penalty=lam, random_state=RANDOM_SEED)
    try:
        jm.fit(features.values)
        states = jm.labels_.astype(int)
    except Exception:
        return np.zeros(len(features), dtype=int)

    if len(np.unique(states)) > 1:
        mean_0 = float(ex_returns.values[states == 0].mean()) if np.any(states == 0) else -np.inf
        mean_1 = float(ex_returns.values[states == 1].mean()) if np.any(states == 1) else -np.inf
        if mean_1 > mean_0:
            states = 1 - states
    return states


def smooth_probabilities(values: np.ndarray, halflife: int) -> np.ndarray:
    if halflife <= 0 or len(values) == 0:
        return values
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def fit_asset_at_refit(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    refit_date: pd.Timestamp,
    period_dates: pd.DatetimeIndex,
    lam: float,
    lookback_years: int = TRAIN_LOOKBACK_YEARS,
) -> tuple[pd.Series, pd.Series]:
    if len(period_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    lookback_days = lookback_years * 252
    position = ex_returns_asset.index.searchsorted(refit_date)
    train_start = max(0, position - lookback_days)
    train_exr = ex_returns_asset.iloc[train_start:position].dropna()
    if len(train_exr) < MIN_TRAIN_OBS:
        fallback = pd.Series(0.5, index=period_dates, name=asset_name)
        return fallback, pd.Series(dtype=float)

    jm_features = build_jm_features(train_exr, asset_name)
    states = fit_jm_sorted(jm_features, train_exr, lam)
    state_series = pd.Series(states, index=train_exr.index, name=asset_name)

    x_series = ex_returns_asset.iloc[:position].dropna()
    x_features = build_gbdt_features(x_series, macro_features.iloc[:position])
    shifted_labels = state_series.shift(-1).dropna()
    common_index = x_features.index.intersection(shifted_labels.index)

    if len(common_index) < 100 or len(np.unique(shifted_labels.reindex(common_index).values)) < 2:
        last_state = int(state_series.iloc[-1]) if not state_series.empty else 0
        fallback_probability = 0.8 if last_state == 0 else 0.2
        fallback = pd.Series(fallback_probability, index=period_dates, name=asset_name)
        return fallback, state_series

    classifier = XGBClassifier(
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
        eval_metric="logloss",
    )
    try:
        classifier.fit(
            x_features.reindex(common_index).fillna(0.0).values,
            shifted_labels.reindex(common_index).values.astype(int),
        )
    except Exception:
        last_state = int(state_series.iloc[-1]) if not state_series.empty else 0
        fallback_probability = 0.8 if last_state == 0 else 0.2
        fallback = pd.Series(fallback_probability, index=period_dates, name=asset_name)
        return fallback, state_series

    ex_oos = ex_returns_asset.loc[:period_dates[-1]].dropna()
    x_oos = build_gbdt_features(ex_oos, macro_features.loc[:period_dates[-1]])
    x_period = x_oos.shift(1).reindex(period_dates).fillna(0.0)
    try:
        probabilities = classifier.predict_proba(x_period.values)[:, 0]
    except Exception:
        probabilities = np.full(len(period_dates), 0.5)

    probabilities = smooth_probabilities(probabilities, GBDT_SMOOTHING_HL.get(asset_name, 0))
    probability_series = pd.Series(probabilities, index=period_dates, name=asset_name)
    return probability_series, state_series


def run_algorithm1_window(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lam: float,
) -> pd.Series:
    window_index = ex_returns_asset.loc[start_date:end_date].index
    if len(window_index) == 0:
        return pd.Series(dtype=float)

    probabilities = pd.Series(np.nan, index=window_index, dtype=float, name=asset_name)
    refit_dates = generate_refit_dates(start_date, end_date, REFIT_MONTHS)
    for idx, refit_date in enumerate(refit_dates):
        next_refit = refit_dates[idx + 1] if idx + 1 < len(refit_dates) else end_date + pd.offsets.BDay(1)
        period_dates = window_index[(window_index >= refit_date) & (window_index < next_refit)]
        if len(period_dates) == 0:
            continue
        period_probabilities, _train_states = fit_asset_at_refit(
            asset_name,
            ex_returns_asset,
            macro_features,
            refit_date,
            period_dates,
            lam,
        )
        probabilities.loc[period_dates] = period_probabilities.reindex(period_dates).values
    return probabilities.ffill(limit=10).fillna(0.5)


def zero_one_strategy_sharpe(probabilities: pd.Series, ex_returns_asset: pd.Series) -> float:
    index = probabilities.dropna().index.intersection(ex_returns_asset.index)
    if len(index) < 50:
        return -99.0
    probability = probabilities.reindex(index).fillna(0.5)
    asset_excess = ex_returns_asset.reindex(index).fillna(0.0)
    strategy = np.where(probability.values > 0.5, asset_excess.values, 0.0)
    mean = float(np.nanmean(strategy) * 252.0)
    std = float(np.nanstd(strategy) * np.sqrt(252.0))
    return mean / (std + 1e-8)


def tuning_cache_file(cache_dir: Path, asset_name: str, refit_date: pd.Timestamp, lam: float) -> Path:
    safe_asset = asset_name.lower()
    return cache_dir / "tuning" / safe_asset / f"{refit_date:%Y%m%d}_{lam:.8f}.json"


def evaluate_lambda_candidate(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    refit_date: pd.Timestamp,
    lam: float,
    cache_dir: Path,
) -> float:
    cache_file = tuning_cache_file(cache_dir, asset_name, refit_date, lam)
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return float(payload["validation_sharpe"])

    validation_start = refit_date - relativedelta(years=VALIDATION_LOOKBACK_YEARS)
    validation_end = refit_date - pd.offsets.BDay(1)
    probabilities = run_algorithm1_window(
        asset_name,
        ex_returns_asset,
        macro_features,
        validation_start,
        validation_end,
        lam,
    )
    sharpe = zero_one_strategy_sharpe(probabilities, ex_returns_asset.loc[validation_start:validation_end])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"validation_sharpe": sharpe, "lambda": lam}, indent=2) + "\n", encoding="utf-8")
    return sharpe


def select_lambda_for_refit(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    refit_date: pd.Timestamp,
    cache_dir: Path,
) -> tuple[float, float]:
    validation_slice = ex_returns_asset.loc[
        refit_date - relativedelta(years=VALIDATION_LOOKBACK_YEARS) : refit_date - pd.offsets.BDay(1)
    ].dropna()
    if len(validation_slice) < MIN_TRAIN_OBS:
        fallback = DEFAULT_LAMBDAS.get(asset_name, LAMBDA_GRID[len(LAMBDA_GRID) // 2])
        return fallback, float("nan")

    best_lambda = DEFAULT_LAMBDAS.get(asset_name, LAMBDA_GRID[0])
    best_sharpe = -np.inf
    for lam in LAMBDA_GRID:
        sharpe = evaluate_lambda_candidate(asset_name, ex_returns_asset, macro_features, refit_date, lam, cache_dir)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_lambda = lam

    if not np.isfinite(best_sharpe):
        best_lambda = DEFAULT_LAMBDAS.get(asset_name, best_lambda)
    return float(best_lambda), float(best_sharpe)


def process_asset_refit(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    refit_date: pd.Timestamp,
    next_refit_date: pd.Timestamp,
    cache_dir: Path,
) -> AssetRefitResult:
    period_end = next_refit_date - pd.offsets.BDay(1)
    period_dates = ex_returns_asset.loc[refit_date:period_end].index
    if len(period_dates) == 0:
        return AssetRefitResult(
            asset_name,
            DEFAULT_LAMBDAS.get(asset_name, 1.0),
            float("nan"),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    best_lambda, validation_sharpe = select_lambda_for_refit(asset_name, ex_returns_asset, macro_features, refit_date, cache_dir)
    probabilities, train_states = fit_asset_at_refit(asset_name, ex_returns_asset, macro_features, refit_date, period_dates, best_lambda)
    return AssetRefitResult(asset_name, best_lambda, validation_sharpe, probabilities, train_states)


def seed_lambda_for_refit(seed_map: SeedLambdaMap | None, asset_name: str, refit_date: pd.Timestamp) -> float | None:
    if not seed_map or asset_name not in seed_map:
        return None
    series = seed_map[asset_name].sort_index()
    eligible = series[series.index <= refit_date]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def process_asset_refit_with_optional_seed(
    asset_name: str,
    ex_returns_asset: pd.Series,
    macro_features: pd.DataFrame,
    refit_date: pd.Timestamp,
    next_refit_date: pd.Timestamp,
    cache_dir: Path,
    seed_map: SeedLambdaMap | None,
) -> AssetRefitResult:
    seeded_lambda = seed_lambda_for_refit(seed_map, asset_name, refit_date)
    if seeded_lambda is not None:
        period_end = next_refit_date - pd.offsets.BDay(1)
        period_dates = ex_returns_asset.loc[refit_date:period_end].index
        probabilities, train_states = fit_asset_at_refit(asset_name, ex_returns_asset, macro_features, refit_date, period_dates, seeded_lambda)
        return AssetRefitResult(asset_name, seeded_lambda, float("nan"), probabilities, train_states)
    return process_asset_refit(asset_name, ex_returns_asset, macro_features, refit_date, next_refit_date, cache_dir)


def ledoit_wolf_shrink(covariance: np.ndarray, n_obs: int) -> np.ndarray:
    n_assets = covariance.shape[0]
    trace_cov = np.trace(covariance)
    trace_cov_sq = np.trace(covariance @ covariance)
    mean_variance = trace_cov / n_assets
    numerator = (n_assets + 2) / max(n_obs, 1)
    denominator = trace_cov_sq / max(trace_cov**2, 1e-10) + (n_assets - 1) / max(n_obs, 1)
    delta = min(1.0, numerator / (denominator + 1e-10))
    shrunk = (1.0 - delta) * covariance + delta * mean_variance * np.eye(n_assets)
    min_eigenvalue = float(np.linalg.eigvalsh(shrunk).min())
    if min_eigenvalue < 1e-6:
        shrunk += (1e-6 - min_eigenvalue) * np.eye(n_assets)
    return shrunk


def estimate_covariance(returns: pd.DataFrame, window: int = COV_WINDOW_DAYS) -> np.ndarray:
    sample = returns.tail(window).dropna(how="any")
    if len(sample) < 30:
        return np.eye(returns.shape[1]) * 0.04
    covariance = sample.cov().values * 252.0
    return ledoit_wolf_shrink(covariance, len(sample))


def estimate_regime_covariance(returns: pd.DataFrame, states: pd.Series, target_state: int, fallback: np.ndarray) -> np.ndarray:
    sample = returns[states.reindex(returns.index) == target_state].dropna(how="any")
    if len(sample) < 30:
        return fallback
    covariance = sample.cov().values * 252.0
    return ledoit_wolf_shrink(covariance, len(sample))


def solve_minvar_with_tc(covariance: np.ndarray, previous_weights: np.ndarray) -> np.ndarray:
    n_assets = covariance.shape[0]
    weights = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(weights, covariance) + GAMMA_TRADE * TC_ONE_WAY * cp.norm1(weights - previous_weights))
    constraints = [cp.sum(weights) == 1.0, weights >= 0.0, weights <= MAX_WEIGHT]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
        if weights.value is not None and np.all(np.isfinite(weights.value)):
            return np.clip(weights.value, 0.0, MAX_WEIGHT)
    except Exception:
        pass
    return np.ones(n_assets) / n_assets


def solve_mv_with_tc(expected_returns: np.ndarray, covariance: np.ndarray, previous_weights: np.ndarray) -> np.ndarray:
    n_assets = covariance.shape[0]
    weights = cp.Variable(n_assets)
    objective = cp.Minimize(
        GAMMA_RISK * cp.quad_form(weights, covariance)
        - expected_returns @ weights
        + GAMMA_TRADE * TC_ONE_WAY * cp.norm1(weights - previous_weights)
    )
    constraints = [cp.sum(weights) <= 1.0, weights >= 0.0, weights <= MAX_WEIGHT]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
        if weights.value is not None and np.all(np.isfinite(weights.value)) and weights.value.sum() > 0.05:
            return np.clip(weights.value, 0.0, MAX_WEIGHT)
    except Exception:
        pass
    return solve_minvar_with_tc(covariance, previous_weights)


def majority_regime_state(asset_states: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    if not asset_states:
        return pd.Series(0, index=index, dtype=float)

    frame = pd.DataFrame({asset: states.reindex(index).ffill(limit=5) for asset, states in asset_states.items()}, index=index)
    bull_fraction = (frame == 0).mean(axis=1)
    return pd.Series(np.where(bull_fraction >= 0.5, 0, 1), index=index, dtype=float)


def regime_mean_return(asset_returns: pd.Series, asset_states: pd.Series, target_state: int, fallback: float) -> float:
    aligned_states = asset_states.reindex(asset_returns.index).ffill(limit=5)
    sample = asset_returns[aligned_states == target_state].dropna().tail(RETURN_ESTIMATION_WINDOW_DAYS)
    if len(sample) < 30:
        return fallback
    return float(sample.mean() * 252.0)


def compute_portfolio_stats(port_returns: pd.Series, rf_daily: pd.Series, label: str) -> dict[str, float | str]:
    returns = port_returns.fillna(0.0)
    rf_aligned = rf_daily.reindex(returns.index).fillna(0.0)
    excess = returns - rf_aligned
    ann_return = float(returns.mean() * 252.0)
    ann_vol = float(returns.std() * np.sqrt(252.0))
    sharpe = ann_return if ann_vol == 0 else float(excess.mean() * 252.0 / (excess.std() * np.sqrt(252.0) + 1e-8))
    wealth = (1.0 + returns).cumprod()
    max_drawdown = float(((wealth / wealth.cummax()) - 1.0).min())
    calmar = ann_return / (abs(max_drawdown) + 1e-8) if ann_return > 0 else np.nan
    return {
        "Label": label,
        "Sharpe": round(sharpe, 3),
        "Ann. Return (%)": round(ann_return * 100.0, 2),
        "Ann. Vol (%)": round(ann_vol * 100.0, 2),
        "Max DD (%)": round(max_drawdown * 100.0, 2),
        "Calmar": round(calmar, 3),
    }


def compute_strategy_returns(
    weights: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    rf_daily: pd.Series,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    strategy_returns: dict[str, pd.Series] = {}
    turnover_rows: list[dict[str, object]] = []

    for strategy_name, weight_frame in weights.items():
        aligned_weights = weight_frame.fillna(0.0)
        dates = aligned_weights.index
        asset_return_frame = asset_returns.reindex(columns=aligned_weights.columns).reindex(dates).fillna(0.0)
        rf_aligned = rf_daily.reindex(dates).ffill().fillna(0.0)

        returns = np.zeros(len(dates))
        turnover = np.zeros(len(dates))
        previous_weights = np.zeros(aligned_weights.shape[1])
        previous_cash = 1.0

        for idx, (_, row) in enumerate(aligned_weights.iterrows()):
            current_weights = row.values.astype(float)
            cash_weight = max(0.0, 1.0 - current_weights.sum())
            trade_size = (np.abs(current_weights - previous_weights).sum() + abs(cash_weight - previous_cash)) / 2.0
            turnover[idx] = trade_size
            gross_return = float(current_weights @ asset_return_frame.iloc[idx].values) + cash_weight * float(rf_aligned.iloc[idx])
            returns[idx] = gross_return - trade_size * TC_ONE_WAY
            previous_weights = current_weights
            previous_cash = cash_weight

        strategy_returns[strategy_name] = pd.Series(returns, index=dates, name=strategy_name)
        turnover_rows.append(
            {
                "Strategy": strategy_name,
                "Avg Daily Turnover": round(float(np.nanmean(turnover)), 6),
                "Annualized Turnover": round(float(np.nanmean(turnover) * 252.0), 3),
            }
        )

    return strategy_returns, pd.DataFrame(turnover_rows)


def build_validation_summary(stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_name, paper_metrics in PAPER_BENCHMARKS.items():
        observed = stats.loc[strategy_name]
        for metric_name in ["Sharpe", "Ann. Return (%)", "Max DD (%)"]:
            paper_value = paper_metrics[metric_name]
            observed_value = float(observed[metric_name])
            delta = observed_value - paper_value
            abs_delta = abs(delta)
            status = "MATCH" if abs_delta < 0.10 else ("CLOSE" if abs_delta < 0.30 else "GAP")
            rows.append(
                {
                    "Strategy": strategy_name,
                    "Metric": metric_name,
                    "Paper": paper_value,
                    "Observed": observed_value,
                    "Delta": round(delta, 3),
                    "Abs Delta": round(abs_delta, 3),
                    "Status": status,
                }
            )
    return pd.DataFrame(rows)


def format_stats_for_console(stats: pd.DataFrame) -> pd.DataFrame:
    formatted = stats.copy()
    for column in ["Ann. Return (%)", "Ann. Vol (%)", "Max DD (%)"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    formatted["Sharpe"] = formatted["Sharpe"].map(lambda value: f"{value:.3f}")
    formatted["Calmar"] = formatted["Calmar"].map(lambda value: f"{value:.3f}" if pd.notna(value) else "nan")
    return formatted


def run_model(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ex_returns: pd.DataFrame,
    macro: pd.DataFrame,
    rf_daily: pd.Series,
    *,
    cache_dir: Path,
    seed_lambda_map: SeedLambdaMap | None = None,
    max_workers: int = DEFAULT_WORKERS,
) -> ModelRun:
    assets = list(ASSETS.keys())
    oos_dates = ex_returns.loc[OOS_START:OOS_END].index
    refit_dates = generate_refit_dates(OOS_START, OOS_END, REFIT_MONTHS)
    macro_features = build_macro_features(macro, returns["LargeCap"], returns["AggBond"])

    weight_history = {
        strategy: pd.DataFrame(index=oos_dates, columns=assets, dtype=float)
        for strategy in ["EW", "EW+Regime", "MinVar", "MinVar+Regime", "MV", "MV+Regime"]
    }
    previous_weights = {strategy: np.zeros(len(assets)) for strategy in weight_history}
    regime_probabilities = {asset: pd.Series(np.nan, index=oos_dates, dtype=float, name=asset) for asset in assets}
    lambda_rows: list[dict[str, object]] = []

    for refit_index, refit_date in enumerate(refit_dates):
        next_refit = refit_dates[refit_index + 1] if refit_index + 1 < len(refit_dates) else pd.Timestamp(OOS_END) + relativedelta(days=1)
        period_dates = oos_dates[(oos_dates >= refit_date) & (oos_dates < next_refit)]
        if len(period_dates) == 0:
            continue

        position = ex_returns.index.searchsorted(refit_date)
        train_start = max(0, position - TRAIN_LOOKBACK_YEARS * 252)
        train_returns = returns.iloc[train_start:position]
        if len(train_returns) < MIN_TRAIN_OBS:
            continue

        print(
            f"Refit {refit_index + 1:02d}/{len(refit_dates)}: {refit_date.date()} "
            f"[train={len(train_returns)}d, oos={len(period_dates)}d]"
        )

        asset_results: dict[str, AssetRefitResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(assets)))) as executor:
            futures = {
                executor.submit(
                    process_asset_refit_with_optional_seed,
                    asset_name,
                    ex_returns[asset_name],
                    macro_features,
                    refit_date,
                    next_refit,
                    cache_dir,
                    seed_lambda_map,
                ): asset_name
                for asset_name in assets
            }
            for future in as_completed(futures):
                result = future.result()
                asset_results[result.asset] = result

        asset_states: dict[str, pd.Series] = {}
        for asset_name in assets:
            result = asset_results[asset_name]
            lambda_rows.append(
                {
                    "Refit Date": refit_date.date().isoformat(),
                    "Asset": asset_name,
                    "Best Lambda": round(result.best_lambda, 6),
                    "Validation Sharpe": round(result.validation_sharpe, 6) if np.isfinite(result.validation_sharpe) else np.nan,
                }
            )
            period_probability = result.probabilities.reindex(period_dates).ffill(limit=10).fillna(0.5)
            regime_probabilities[asset_name].loc[period_dates] = period_probability.values
            asset_states[asset_name] = result.train_states

        aggregate_states = majority_regime_state(asset_states, train_returns.index)
        sigma_full = estimate_covariance(train_returns)
        sigma_bull = estimate_regime_covariance(train_returns, aggregate_states, 0, sigma_full)
        sigma_bear = estimate_regime_covariance(train_returns, aggregate_states, 1, sigma_full)

        win = min(RETURN_ESTIMATION_WINDOW_DAYS, len(train_returns))
        mu_base = train_returns.tail(win).mean().values * 252.0
        mu_base = 0.5 * mu_base + 0.5 * mu_base.mean()

        mu_bull = np.array(
            [
                regime_mean_return(train_returns[asset], asset_states.get(asset, pd.Series(dtype=float)), 0, mu_base[idx])
                for idx, asset in enumerate(assets)
            ]
        )
        mu_bear = np.array(
            [
                regime_mean_return(train_returns[asset], asset_states.get(asset, pd.Series(dtype=float)), 1, mu_base[idx] * 0.3)
                for idx, asset in enumerate(assets)
            ]
        )
        mu_bull = np.clip(0.5 * mu_bull + 0.5 * mu_bull.mean(), -0.5, 0.5)
        mu_bear = np.clip(0.5 * mu_bear + 0.5 * mu_bear.mean(), -0.5, 0.5)

        baseline_ew = np.ones(len(assets)) / len(assets)
        baseline_minvar = solve_minvar_with_tc(sigma_full, previous_weights["MinVar"])
        baseline_mv = solve_mv_with_tc(mu_base, sigma_full, previous_weights["MV"])

        for period_date in period_dates:
            probabilities = np.array(
                [float(regime_probabilities[asset].loc[period_date]) for asset in assets],
                dtype=float,
            ).clip(0.0, 1.0)
            average_bull_probability = float(probabilities.mean())
            sigma_regime = average_bull_probability * sigma_bull + (1.0 - average_bull_probability) * sigma_bear
            min_eigenvalue = float(np.linalg.eigvalsh(sigma_regime).min())
            if min_eigenvalue < 1e-6:
                sigma_regime += (1e-6 - min_eigenvalue) * np.eye(len(assets))

            ew_regime_mask = probabilities > 0.5
            ew_regime_weights = np.zeros(len(assets))
            if ew_regime_mask.sum() > 0:
                ew_regime_weights[ew_regime_mask] = 1.0 / ew_regime_mask.sum()

            mv_regime_expected = probabilities * mu_bull + (1.0 - probabilities) * mu_bear

            weight_history["EW"].loc[period_date] = baseline_ew
            weight_history["EW+Regime"].loc[period_date] = ew_regime_weights
            weight_history["MinVar"].loc[period_date] = baseline_minvar
            weight_history["MV"].loc[period_date] = baseline_mv
            weight_history["MinVar+Regime"].loc[period_date] = solve_minvar_with_tc(sigma_regime, previous_weights["MinVar+Regime"])
            weight_history["MV+Regime"].loc[period_date] = solve_mv_with_tc(mv_regime_expected, sigma_regime, previous_weights["MV+Regime"])

            for strategy_name in weight_history:
                previous_weights[strategy_name] = weight_history[strategy_name].loc[period_date].values.astype(float)

    strategy_returns, turnover_summary = compute_strategy_returns(weight_history, returns, rf_daily.reindex(oos_dates))
    stats_rows = [compute_portfolio_stats(strategy_returns[strategy], rf_daily.reindex(oos_dates), strategy) for strategy in strategy_returns]
    stats = pd.DataFrame(stats_rows).set_index("Label")
    stats = stats.loc[["EW", "EW+Regime", "MinVar", "MinVar+Regime", "MV", "MV+Regime"]]
    validation_summary = build_validation_summary(stats)
    lambda_history = pd.DataFrame(lambda_rows)
    return ModelRun(
        strategy_returns=strategy_returns,
        stats=stats,
        validation_summary=validation_summary,
        turnover_summary=turnover_summary,
        lambda_history=lambda_history,
        regime_probabilities=regime_probabilities,
        weights=weight_history,
        data_coverage=build_data_coverage_table(prices),
        source_mapping=build_source_mapping_table(ASSETS, PROXY_FUNDS),
    )


def save_artifacts(run: ModelRun, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run.stats.to_csv(artifacts_dir / "performance_stats.csv")
    run.validation_summary.to_csv(artifacts_dir / "validation_summary.csv", index=False)
    run.turnover_summary.to_csv(artifacts_dir / "turnover_summary.csv", index=False)
    run.lambda_history.to_csv(artifacts_dir / "lambda_history.csv", index=False)
    run.data_coverage.to_csv(artifacts_dir / "data_coverage.csv", index=False)
    run.source_mapping.to_csv(artifacts_dir / "source_mapping.csv", index=False)

    strategy_returns_frame = pd.DataFrame(run.strategy_returns)
    strategy_returns_frame.to_csv(artifacts_dir / "daily_strategy_returns.csv")

    probability_frame = pd.DataFrame(run.regime_probabilities)
    probability_frame.to_csv(artifacts_dir / "regime_probabilities.csv")

    cumulative = (1.0 + strategy_returns_frame).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative.plot(ax=ax, linewidth=1.7)
    ax.set_title("Cumulative Net Returns, 2007-2023")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_returns.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Paper": {strategy: PAPER_BENCHMARKS[strategy]["Sharpe"] for strategy in run.stats.index},
            "Observed": {strategy: float(run.stats.loc[strategy, "Sharpe"]) for strategy in run.stats.index},
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Sharpe Ratio Comparison")
    ax.set_ylabel("Sharpe")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "sharpe_comparison.png", dpi=160)
    plt.close(fig)

    lambda_pivot = run.lambda_history.pivot(index="Asset", columns="Refit Date", values="Best Lambda")
    fig, ax = plt.subplots(figsize=(12, 5))
    heatmap = ax.imshow(np.log10(lambda_pivot.values.astype(float)), aspect="auto", cmap="viridis")
    ax.set_title("Rolling Lambda Selection (log10 scale)")
    ax.set_yticks(range(len(lambda_pivot.index)))
    ax.set_yticklabels(lambda_pivot.index)
    ax.set_xticks(range(len(lambda_pivot.columns)))
    ax.set_xticklabels(lambda_pivot.columns, rotation=45, ha="right")
    fig.colorbar(heatmap, ax=ax, label="log10(lambda)")
    fig.tight_layout()
    fig.savefig(figures_dir / "lambda_heatmap.png", dpi=160)
    plt.close(fig)

    selected_assets = [asset for asset in ["LargeCap", "AggBond", "HighYield", "Gold"] if asset in probability_frame.columns]
    fig, axes = plt.subplots(len(selected_assets), 1, figsize=(10, 2.4 * max(1, len(selected_assets))), sharex=True)
    if len(selected_assets) == 1:
        axes = [axes]
    for axis, asset_name in zip(axes, selected_assets):
        axis.plot(probability_frame.index, probability_frame[asset_name], linewidth=1.2)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel(asset_name)
        axis.grid(alpha=0.25)
    axes[0].set_title("Selected Bull Probabilities")
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_probabilities.png", dpi=160)
    plt.close(fig)
