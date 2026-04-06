from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import factorized

from data_access import KwonRawData


FACTOR_ORDER = ["Size", "Value", "Momentum", "Profitability", "Investment"]
REGIME_ORDER = ["Recovery", "Expansion", "Slowdown", "Contraction"]

PAPER_TABLE1_COUNTS = {
    "Recovery": 172,
    "Expansion": 215,
    "Slowdown": 168,
    "Contraction": 102,
}

PAPER_TABLE1_TRANSITIONS = pd.DataFrame(
    [
        [0.94, 0.05, 0.00, 0.01],
        [0.00, 0.93, 0.07, 0.00],
        [0.00, 0.03, 0.92, 0.05],
        [0.10, 0.00, 0.00, 0.90],
    ],
    index=REGIME_ORDER,
    columns=REGIME_ORDER,
)

PAPER_TABLE2 = {
    "Ann Return (%)": {"Size": 2.622, "Value": 3.165, "Momentum": 7.312, "Profitability": 3.328, "Investment": 3.311},
    "Ann Vol (%)": {"Size": 10.627, "Value": 10.273, "Momentum": 14.879, "Profitability": 7.751, "Investment": 6.906},
    "Sharpe": {"Size": 0.247, "Value": 0.308, "Momentum": 0.491, "Profitability": 0.429, "Investment": 0.479},
    "Skew": {"Size": 0.371, "Value": 0.016, "Momentum": -1.273, "Profitability": -0.358, "Investment": 0.369},
    "Excess Kurtosis": {"Size": 0.182, "Value": -1.157, "Momentum": 6.627, "Profitability": 8.987, "Investment": -1.790},
}

PAPER_TABLE3 = {
    "Recovery": {
        "Ann Return (%)": {"Size": 10.319, "Value": 0.977, "Momentum": -3.616, "Profitability": 0.210, "Investment": 1.808},
        "Ann Vol (%)": {"Size": 9.241, "Value": 9.954, "Momentum": 17.096, "Profitability": 7.515, "Investment": 6.589},
        "Sharpe": {"Size": 1.117, "Value": 0.098, "Momentum": -0.212, "Profitability": 0.028, "Investment": 0.274},
    },
    "Expansion": {
        "Ann Return (%)": {"Size": 6.143, "Value": 4.299, "Momentum": 14.744, "Profitability": 0.974, "Investment": 1.555},
        "Ann Vol (%)": {"Size": 9.512, "Value": 9.119, "Momentum": 10.854, "Profitability": 6.005, "Investment": 6.240},
        "Sharpe": {"Size": 0.645, "Value": 0.471, "Momentum": 1.358, "Profitability": 0.162, "Investment": 0.249},
    },
    "Slowdown": {
        "Ann Return (%)": {"Size": -5.612, "Value": 2.457, "Momentum": 7.128, "Profitability": 3.933, "Investment": 3.128},
        "Ann Vol (%)": {"Size": 11.666, "Value": 9.328, "Momentum": 13.283, "Profitability": 8.368, "Investment": 6.241},
        "Sharpe": {"Size": -0.481, "Value": 0.263, "Momentum": 0.537, "Profitability": 0.470, "Investment": 0.501},
    },
    "Contraction": {
        "Ann Return (%)": {"Size": -4.018, "Value": 5.357, "Momentum": 13.946, "Profitability": 11.461, "Investment": 9.503},
        "Ann Vol (%)": {"Size": 11.588, "Value": 13.689, "Momentum": 18.608, "Profitability": 9.429, "Investment": 8.966},
        "Sharpe": {"Size": -0.347, "Value": 0.391, "Momentum": 0.749, "Profitability": 1.216, "Investment": 1.060},
    },
}

PAPER_TABLE4 = {
    "Benchmark": {"Size": 23.1, "Value": 14.5, "Momentum": 13.5, "Profitability": 27.5, "Investment": 21.4},
    "Recovery": {"Size": 79.5, "Value": 0.0, "Momentum": 0.0, "Profitability": 11.1, "Investment": 9.4},
    "Expansion": {"Size": 23.5, "Value": 22.1, "Momentum": 54.4, "Profitability": 0.0, "Investment": 0.0},
    "Slowdown": {"Size": 0.0, "Value": 3.8, "Momentum": 27.3, "Profitability": 34.5, "Investment": 34.4},
    "Contraction": {"Size": 0.0, "Value": 0.0, "Momentum": 25.9, "Profitability": 46.1, "Investment": 28.0},
}

PAPER_TABLE5 = {
    "Benchmark Ann Return (%)": 0.444,
    "Benchmark Ann Vol (%)": 3.485,
    "Benchmark Sharpe": 0.127,
    "Benchmark Max DD (%)": -13.979,
    "Benchmark Turnover (%)": 1.673,
    "Dynamic Ann Return (%)": 5.042,
    "Dynamic Ann Vol (%)": 6.055,
    "Dynamic Sharpe": 0.833,
    "Dynamic Max DD (%)": -12.566,
    "Dynamic Turnover (%)": 29.607,
    "Tracking Error (%)": 5.532,
    "Information Ratio": 0.626,
    "Break-even TC (%)": 1.202,
}

PAPER_TABLE6 = {
    0.3: {"Ann Return (%)": 3.185, "Sharpe": 0.640},
    0.5: {"Ann Return (%)": 5.042, "Sharpe": 0.833},
    0.7: {"Ann Return (%)": 7.141, "Sharpe": 0.984},
}

PAPER_TABLE7 = {
    0.3: {"Sharpe": 1.096},
    0.5: {"Sharpe": 0.833},
    0.7: {"Sharpe": 0.538},
}

PAPER_TABLE8 = {
    3.0: {"Sharpe": 1.035},
    5.0: {"Sharpe": 0.833},
    7.0: {"Sharpe": 0.732},
}


@dataclass(frozen=True)
class KwonConfig:
    data_end: str = "2021-10-31"
    fred_vintage_date: str | None = "2021-10-31"
    lambda_value: float = 0.5
    kappa: float = 0.5
    delta: float = 5.0
    slope_method: str = "center"
    oos_start: str = "2007-01-01"
    oos_end: str = "2021-10-01"
    in_sample_end: str = "2006-12-01"
    benchmark_window: int | None = None


@dataclass
class PreparedData:
    raw_macro: pd.DataFrame
    standardized_macro: pd.DataFrame
    macro_indicator: pd.Series
    macro_mean: pd.Series
    macro_std: pd.Series
    pc1_loadings: pd.Series
    pc1_explained_variance_ratio: float
    factors: pd.DataFrame
    vix_proxy: pd.Series
    vix_spliced: pd.Series


@dataclass
class FullSampleRegimeResult:
    regimes: pd.Series
    trend: pd.Series
    slope: pd.Series
    counts: pd.Series
    transition_matrix: pd.DataFrame


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights_rp: pd.DataFrame
    weights_dynamic: pd.DataFrame
    forecast_regimes: pd.Series
    summary: pd.DataFrame


@dataclass
class BaselineAnalysis:
    config: KwonConfig
    prepared: PreparedData
    full_sample_regimes: FullSampleRegimeResult
    factor_descriptive_stats: pd.DataFrame
    factor_correlation_matrix: pd.DataFrame
    regime_factor_stats: pd.DataFrame
    in_sample_allocations: pd.DataFrame
    backtest: BacktestResult


def month_start_index(index: Iterable[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).to_period("M").to_timestamp()


def monthly_last(series: pd.Series) -> pd.Series:
    values = series.dropna().sort_index().resample("ME").last().dropna()
    values.index = month_start_index(values.index)
    return values


def monthly_mean(series: pd.Series) -> pd.Series:
    values = series.dropna().sort_index().resample("ME").mean().dropna()
    values.index = month_start_index(values.index)
    return values


def compute_vix_proxy(sp500_close: pd.Series) -> pd.Series:
    returns = sp500_close.sort_index().pct_change().dropna()
    proxy = returns.resample("ME").std().dropna() * np.sqrt(21.0) * 100.0
    proxy.index = month_start_index(proxy.index)
    return proxy.rename("vix_proxy")


def build_spliced_vix(vixcls: pd.Series, sp500_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    proxy = compute_vix_proxy(sp500_close)
    vix_monthly = monthly_mean(vixcls).rename("vixcls")
    splice_date = pd.Timestamp("1990-01-01")
    spliced = pd.concat([proxy[proxy.index < splice_date], vix_monthly[vix_monthly.index >= splice_date]]).sort_index()
    spliced = spliced[~spliced.index.duplicated(keep="last")]
    return proxy, spliced.rename("vix_spliced")


def prepare_data(raw_data: KwonRawData) -> PreparedData:
    gs10_m = monthly_last(raw_data.gs10)
    fedfunds_m = monthly_last(raw_data.fedfunds)
    baa_m = monthly_last(raw_data.baa)
    ic4wsa_m = monthly_last(raw_data.ic4wsa)
    permit_m = monthly_last(raw_data.permit)
    vix_proxy, vix_spliced = build_spliced_vix(raw_data.vixcls, raw_data.sp500_close)

    raw_macro = pd.DataFrame(
        {
            "yield_spread": -(gs10_m - fedfunds_m),
            "credit_spread": -(baa_m - gs10_m),
            "jobless_claims": -ic4wsa_m.shift(1),
            "bldg_permits": permit_m.shift(1),
            "vix": -vix_spliced,
        }
    ).dropna()

    macro_mean = raw_macro.mean()
    macro_std = raw_macro.std().replace(0.0, np.nan)
    standardized = ((raw_macro - macro_mean) / macro_std).dropna()
    scores, loadings, explained_ratio = fit_first_pc(standardized)
    if np.corrcoef(scores, raw_macro.loc[standardized.index, "bldg_permits"])[0, 1] < 0:
        loadings = -loadings
        scores = -scores

    macro_indicator = pd.Series(scores, index=standardized.index, name="macro_indicator")
    factors = raw_data.factors.loc[raw_data.factors.index >= macro_indicator.index.min()].copy()

    return PreparedData(
        raw_macro=raw_macro,
        standardized_macro=standardized,
        macro_indicator=macro_indicator,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pc1_loadings=pd.Series(loadings, index=standardized.columns, name="pc1_loading"),
        pc1_explained_variance_ratio=explained_ratio,
        factors=factors,
        vix_proxy=vix_proxy,
        vix_spliced=vix_spliced,
    )


def fit_first_pc(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = frame.values
    covariance = np.cov(matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    pc1 = eigenvectors[:, 0]
    explained_ratio = float(eigenvalues[0] / eigenvalues.sum())
    scores = matrix @ pc1
    return scores, pc1, explained_ratio


def compute_mi_from_raw(raw_macro: pd.DataFrame, macro_mean: pd.Series, macro_std: pd.Series, pc1_loadings: pd.Series) -> pd.Series:
    standardized = (raw_macro - macro_mean) / macro_std
    standardized = standardized.dropna()
    values = standardized.loc[:, pc1_loadings.index].values @ pc1_loadings.values
    return pd.Series(values, index=standardized.index, name="macro_indicator")


@lru_cache(maxsize=None)
def _l1_system(n: int, rho: float) -> tuple[sparse.csc_matrix, callable]:
    if n < 3:
        raise ValueError("L1 trend filter requires at least 3 observations.")
    ones = np.ones(n - 2)
    difference = sparse.diags([ones, -2.0 * ones, ones], [0, 1, 2], shape=(n - 2, n), format="csc")
    system = sparse.eye(n, format="csc") + rho * (difference.T @ difference)
    return difference, factorized(system)


def l1_trend_filter(
    values: np.ndarray,
    lambda_value: float,
    *,
    rho: float = 1.0,
    max_iter: int = 5000,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-4,
) -> np.ndarray:
    y = np.asarray(values, dtype=float).ravel()
    n = len(y)
    if n < 3:
        return y.copy()

    difference, solve = _l1_system(n, rho)
    x = y.copy()
    z = np.zeros(n - 2)
    u = np.zeros(n - 2)

    for _iteration in range(max_iter):
        rhs = y + rho * (difference.T @ (z - u))
        x = np.asarray(solve(rhs)).ravel()
        dx = np.asarray(difference @ x).ravel()
        z_previous = z.copy()
        z = np.sign(dx + u) * np.maximum(np.abs(dx + u) - (lambda_value / rho), 0.0)
        u = u + dx - z

        primal_residual = np.linalg.norm(dx - z)
        dual_residual = np.linalg.norm(rho * (difference.T @ (z - z_previous)))
        epsilon_primal = np.sqrt(len(z)) * abs_tol + rel_tol * max(np.linalg.norm(dx), np.linalg.norm(z))
        epsilon_dual = np.sqrt(n) * abs_tol + rel_tol * np.linalg.norm(rho * (difference.T @ u))
        if primal_residual <= epsilon_primal and dual_residual <= epsilon_dual:
            break

    return x


def compute_slope(trend: np.ndarray, method: str) -> np.ndarray:
    if method == "forward":
        return np.diff(trend, append=trend[-1])
    if method == "backward":
        return np.diff(trend, prepend=trend[0])
    if method == "center":
        forward = np.diff(trend, append=trend[-1])
        backward = np.diff(trend, prepend=trend[0])
        return 0.5 * (forward + backward)
    raise ValueError(f"Unsupported slope method: {method}")


def classify_regime(level_value: float, slope_value: float) -> str:
    above_average = level_value >= 0.0
    accelerating = slope_value > 0.0
    if not above_average and accelerating:
        return "Recovery"
    if above_average and accelerating:
        return "Expansion"
    if above_average and not accelerating:
        return "Slowdown"
    return "Contraction"


def compute_regimes(macro_indicator: pd.Series, *, lambda_value: float, slope_method: str) -> FullSampleRegimeResult:
    trend_values = l1_trend_filter(macro_indicator.values, lambda_value=lambda_value)
    slope_values = compute_slope(trend_values, slope_method)
    regimes = pd.Series(
        [classify_regime(level, slope) for level, slope in zip(macro_indicator.values, slope_values)],
        index=macro_indicator.index,
        name="regime",
    )
    trend = pd.Series(trend_values, index=macro_indicator.index, name="l1_trend")
    slope = pd.Series(slope_values, index=macro_indicator.index, name="slope")
    counts = regimes.value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)
    transition_matrix = compute_transition_matrix(regimes)
    return FullSampleRegimeResult(regimes=regimes, trend=trend, slope=slope, counts=counts, transition_matrix=transition_matrix)


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    counts = pd.DataFrame(0.0, index=REGIME_ORDER, columns=REGIME_ORDER)
    previous = regimes.iloc[:-1].values
    current = regimes.iloc[1:].values
    for left, right in zip(previous, current):
        counts.loc[left, right] += 1.0
    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    return counts.div(row_sums, axis=0).fillna(0.0)


def compute_factor_descriptive_stats(factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors_pct = factors[FACTOR_ORDER] * 100.0
    stats = pd.DataFrame(index=FACTOR_ORDER)
    stats["Ann Return (%)"] = factors_pct.mean() * 12.0
    stats["Ann Vol (%)"] = factors_pct.std() * np.sqrt(12.0)
    stats["Sharpe"] = stats["Ann Return (%)"] / stats["Ann Vol (%)"]
    stats["Skew"] = factors_pct.skew()
    stats["Excess Kurtosis"] = factors_pct.kurtosis()
    correlation = factors_pct.corr()
    return stats, correlation


def compute_regime_factor_stats(factors: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    factors_pct = factors[FACTOR_ORDER] * 100.0
    rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        mask = regimes.reindex(factors_pct.index) == regime
        subset = factors_pct.loc[mask.fillna(False)]
        ann_return = subset.mean() * 12.0
        ann_vol = subset.std() * np.sqrt(12.0)
        sharpe = ann_return / ann_vol
        for factor in FACTOR_ORDER:
            rows.append(
                {
                    "Regime": regime,
                    "Factor": factor,
                    "N": int(len(subset)),
                    "Ann Return (%)": float(ann_return.get(factor, np.nan)),
                    "Ann Vol (%)": float(ann_vol.get(factor, np.nan)),
                    "Sharpe": float(sharpe.get(factor, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def risk_parity_weights(sigma: np.ndarray) -> np.ndarray:
    n_assets = sigma.shape[0]

    def objective(weights: np.ndarray) -> float:
        portfolio_vol = np.sqrt(weights @ sigma @ weights + 1e-16)
        risk_contrib = weights * (sigma @ weights / portfolio_vol)
        return float(np.sum((risk_contrib - risk_contrib.mean()) ** 2))

    result = minimize(
        objective,
        x0=np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    if not result.success:
        weights = np.full(n_assets, 1.0 / n_assets)
    else:
        weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def bl_mvo_weights(sigma: np.ndarray, risk_parity: np.ndarray, q_view: np.ndarray, *, delta: float, kappa: float) -> np.ndarray:
    implied = delta * (sigma @ risk_parity)
    posterior = (1.0 - kappa) * implied + kappa * q_view
    n_assets = len(risk_parity)

    def objective(weights: np.ndarray) -> float:
        utility = posterior @ weights - 0.5 * delta * (weights @ sigma @ weights)
        return float(-utility)

    result = minimize(
        objective,
        x0=risk_parity,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        weights = risk_parity.copy()
    else:
        weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() <= 0:
        return risk_parity.copy()
    return weights / weights.sum()


def ar1_forecast(series: pd.Series) -> float:
    values = np.asarray(series.dropna().values, dtype=float)
    if len(values) < 3:
        return float(values[-1])
    x = np.column_stack([np.ones(len(values) - 1), values[:-1]])
    y = values[1:]
    beta, *_rest = np.linalg.lstsq(x, y, rcond=None)
    forecast = beta[0] + beta[1] * values[-1]
    if not np.isfinite(forecast):
        return float(values[-1])
    return float(forecast)


def run_backtest(
    prepared: PreparedData,
    *,
    lambda_value: float,
    kappa: float,
    delta: float,
    slope_method: str,
    oos_start: str,
    oos_end: str,
    benchmark_window: int | None = None,
) -> BacktestResult:
    factors = prepared.factors[FACTOR_ORDER]
    oos_dates = factors.index[(factors.index >= pd.Timestamp(oos_start)) & (factors.index <= pd.Timestamp(oos_end))]
    results: list[dict[str, object]] = []

    for current_date in oos_dates:
        previous_date = current_date - pd.offsets.MonthBegin(1)
        historical_factors = factors.loc[factors.index <= previous_date]
        historical_macro = prepared.raw_macro.loc[prepared.raw_macro.index <= previous_date]
        if len(historical_factors) < 24 or len(historical_macro) < 24:
            continue

        windowed_factors = historical_factors.iloc[-benchmark_window:] if benchmark_window else historical_factors
        sigma = windowed_factors.cov().values
        risk_parity = risk_parity_weights(sigma)

        mi_history = compute_mi_from_raw(historical_macro, prepared.macro_mean, prepared.macro_std, prepared.pc1_loadings)
        mi_next = ar1_forecast(mi_history)

        extended_values = np.append(mi_history.values, mi_next)
        extended_trend = l1_trend_filter(extended_values, lambda_value=lambda_value)
        extended_slope = compute_slope(extended_trend, slope_method)
        forecast_regime = classify_regime(mi_next, extended_slope[-1])

        historical_trend = extended_trend[:-1]
        historical_slope = compute_slope(historical_trend, slope_method)
        historical_regimes = pd.Series(
            [classify_regime(level, slope) for level, slope in zip(mi_history.values, historical_slope)],
            index=mi_history.index,
        )

        aligned_index = historical_factors.index.intersection(mi_history.index)
        same_regime = historical_regimes.reindex(aligned_index) == forecast_regime
        regime_subset = historical_factors.loc[aligned_index][same_regime.fillna(False).values]
        q_view = regime_subset.mean().values if len(regime_subset) >= 5 else historical_factors.mean().values
        dynamic_weights = bl_mvo_weights(sigma, risk_parity, q_view, delta=delta, kappa=kappa)

        realized_returns = factors.loc[current_date].values
        results.append(
            {
                "Date": current_date,
                "Benchmark": float(risk_parity @ realized_returns),
                "Dynamic": float(dynamic_weights @ realized_returns),
                "Active": float((dynamic_weights - risk_parity) @ realized_returns),
                "Forecast Regime": forecast_regime,
                "Risk Parity Weights": risk_parity.copy(),
                "Dynamic Weights": dynamic_weights.copy(),
            }
        )

    if not results:
        raise RuntimeError("The OOS backtest did not produce any results.")

    index = pd.DatetimeIndex([row["Date"] for row in results])
    returns = pd.DataFrame(
        {
            "Benchmark": [float(row["Benchmark"]) for row in results],
            "Dynamic": [float(row["Dynamic"]) for row in results],
            "Active": [float(row["Active"]) for row in results],
        },
        index=index,
    )
    forecast_regimes = pd.Series([str(row["Forecast Regime"]) for row in results], index=index, name="Forecast Regime")
    weights_rp = pd.DataFrame([row["Risk Parity Weights"] for row in results], index=index, columns=FACTOR_ORDER)
    weights_dynamic = pd.DataFrame([row["Dynamic Weights"] for row in results], index=index, columns=FACTOR_ORDER)
    summary = compute_oos_summary(returns, weights_rp, weights_dynamic)
    return BacktestResult(
        returns=returns,
        weights_rp=weights_rp,
        weights_dynamic=weights_dynamic,
        forecast_regimes=forecast_regimes,
        summary=summary,
    )


def annualized_performance(series: pd.Series) -> dict[str, float]:
    ann_return = float(series.mean() * 12.0 * 100.0)
    ann_vol = float(series.std() * np.sqrt(12.0) * 100.0)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    cumulative = (1.0 + series).cumprod()
    max_drawdown = float(((cumulative / cumulative.cummax()) - 1.0).min() * 100.0)
    return {
        "Ann Return (%)": ann_return,
        "Ann Vol (%)": ann_vol,
        "Sharpe": sharpe,
        "Max DD (%)": max_drawdown,
    }


def compute_turnover(weights: pd.DataFrame) -> float:
    if len(weights) < 2:
        return 0.0
    deltas = np.abs(np.diff(weights.values, axis=0)).sum(axis=1)
    return float(np.mean(deltas) * 100.0)


def compute_oos_summary(returns: pd.DataFrame, weights_rp: pd.DataFrame, weights_dynamic: pd.DataFrame) -> pd.DataFrame:
    bench_stats = annualized_performance(returns["Benchmark"])
    dyn_stats = annualized_performance(returns["Dynamic"])
    benchmark_turnover = compute_turnover(weights_rp)
    dynamic_turnover = compute_turnover(weights_dynamic)
    tracking_error = float(returns["Active"].std() * np.sqrt(12.0) * 100.0)
    information_ratio = (
        (dyn_stats["Ann Return (%)"] - bench_stats["Ann Return (%)"]) / tracking_error if tracking_error else np.nan
    )
    break_even_tc = (
        (dyn_stats["Ann Return (%)"] - bench_stats["Sharpe"] * dyn_stats["Ann Vol (%)"]) / (dynamic_turnover / 100.0 * 12.0)
        if dynamic_turnover
        else np.nan
    )
    rows = [
        ("Benchmark Ann Return (%)", bench_stats["Ann Return (%)"]),
        ("Benchmark Ann Vol (%)", bench_stats["Ann Vol (%)"]),
        ("Benchmark Sharpe", bench_stats["Sharpe"]),
        ("Benchmark Max DD (%)", bench_stats["Max DD (%)"]),
        ("Benchmark Turnover (%)", benchmark_turnover),
        ("Dynamic Ann Return (%)", dyn_stats["Ann Return (%)"]),
        ("Dynamic Ann Vol (%)", dyn_stats["Ann Vol (%)"]),
        ("Dynamic Sharpe", dyn_stats["Sharpe"]),
        ("Dynamic Max DD (%)", dyn_stats["Max DD (%)"]),
        ("Dynamic Turnover (%)", dynamic_turnover),
        ("Tracking Error (%)", tracking_error),
        ("Information Ratio", information_ratio),
        ("Break-even TC (%)", break_even_tc),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Observed"]).set_index("Metric")


def compute_in_sample_allocations(
    prepared: PreparedData,
    full_sample_regimes: FullSampleRegimeResult,
    *,
    in_sample_end: str,
    delta: float,
    kappa: float,
) -> pd.DataFrame:
    factors = prepared.factors[FACTOR_ORDER]
    in_sample_factors = factors.loc[factors.index <= pd.Timestamp(in_sample_end)]
    sigma = in_sample_factors.cov().values
    benchmark = risk_parity_weights(sigma)
    rows = [{"Portfolio": "Benchmark", **{factor: float(weight * 100.0) for factor, weight in zip(FACTOR_ORDER, benchmark)}}]

    for regime in REGIME_ORDER:
        mask = full_sample_regimes.regimes.reindex(in_sample_factors.index) == regime
        regime_subset = in_sample_factors.loc[mask.fillna(False)]
        q_view = regime_subset.mean().values if len(regime_subset) else in_sample_factors.mean().values
        weights = bl_mvo_weights(sigma, benchmark, q_view, delta=delta, kappa=kappa)
        rows.append({"Portfolio": regime, **{factor: float(weight * 100.0) for factor, weight in zip(FACTOR_ORDER, weights)}})
    frame = pd.DataFrame(rows).set_index("Portfolio")
    return frame[FACTOR_ORDER]


def run_baseline_analysis(raw_data: KwonRawData, config: KwonConfig) -> BaselineAnalysis:
    prepared = prepare_data(raw_data)
    full_sample_regimes = compute_regimes(
        prepared.macro_indicator,
        lambda_value=config.lambda_value,
        slope_method=config.slope_method,
    )
    factor_stats, factor_corr = compute_factor_descriptive_stats(prepared.factors)
    regime_factor_stats = compute_regime_factor_stats(prepared.factors, full_sample_regimes.regimes)
    in_sample_allocations = compute_in_sample_allocations(
        prepared,
        full_sample_regimes,
        in_sample_end=config.in_sample_end,
        delta=config.delta,
        kappa=config.kappa,
    )
    backtest = run_backtest(
        prepared,
        lambda_value=config.lambda_value,
        kappa=config.kappa,
        delta=config.delta,
        slope_method=config.slope_method,
        oos_start=config.oos_start,
        oos_end=config.oos_end,
        benchmark_window=config.benchmark_window,
    )
    return BaselineAnalysis(
        config=config,
        prepared=prepared,
        full_sample_regimes=full_sample_regimes,
        factor_descriptive_stats=factor_stats,
        factor_correlation_matrix=factor_corr,
        regime_factor_stats=regime_factor_stats,
        in_sample_allocations=in_sample_allocations,
        backtest=backtest,
    )


def compare_series_to_targets(observed: pd.Series, targets: dict[str, float], metric_name: str) -> pd.DataFrame:
    rows = []
    for key, paper_value in targets.items():
        observed_value = float(observed.get(key, np.nan))
        rows.append(
            {
                "Metric": metric_name,
                "Item": key,
                "Paper": paper_value,
                "Observed": observed_value,
                "Abs Delta": abs(observed_value - paper_value) if np.isfinite(observed_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_table1_validation(result: FullSampleRegimeResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows = []
    for regime in REGIME_ORDER:
        observed = int(result.counts.get(regime, 0))
        paper = PAPER_TABLE1_COUNTS[regime]
        count_rows.append({"Regime": regime, "Paper": paper, "Observed": observed, "Abs Delta": abs(observed - paper)})
    transition_rows = []
    for left in REGIME_ORDER:
        for right in REGIME_ORDER:
            paper = float(PAPER_TABLE1_TRANSITIONS.loc[left, right] * 100.0)
            observed = float(result.transition_matrix.loc[left, right] * 100.0)
            transition_rows.append(
                {
                    "From": left,
                    "To": right,
                    "Paper (%)": paper,
                    "Observed (%)": observed,
                    "Abs Delta": abs(observed - paper),
                }
            )
    return pd.DataFrame(count_rows), pd.DataFrame(transition_rows)


def build_table2_validation(stats: pd.DataFrame) -> pd.DataFrame:
    frames = [compare_series_to_targets(stats[metric], targets, metric) for metric, targets in PAPER_TABLE2.items()]
    return pd.concat(frames, ignore_index=True)


def build_table3_validation(regime_factor_stats: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for regime, metric_targets in PAPER_TABLE3.items():
        subset = regime_factor_stats.loc[regime_factor_stats["Regime"] == regime].set_index("Factor")
        for metric, targets in metric_targets.items():
            comparison = compare_series_to_targets(subset[metric], targets, metric)
            comparison.insert(0, "Regime", regime)
            frames.append(comparison)
    return pd.concat(frames, ignore_index=True)


def build_table4_validation(in_sample_allocations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for portfolio, targets in PAPER_TABLE4.items():
        comparison = compare_series_to_targets(in_sample_allocations.loc[portfolio], targets, "Allocation (%)")
        comparison.insert(0, "Portfolio", portfolio)
        frames.append(comparison)
    return pd.concat(frames, ignore_index=True)


def build_table5_validation(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, paper_value in PAPER_TABLE5.items():
        observed = float(summary.loc[metric, "Observed"])
        rows.append({"Metric": metric, "Paper": paper_value, "Observed": observed, "Abs Delta": abs(observed - paper_value)})
    return pd.DataFrame(rows)


def build_slope_method_comparison(prepared: PreparedData, config: KwonConfig) -> pd.DataFrame:
    rows = []
    for method in ["forward", "backward", "center"]:
        full_sample = compute_regimes(prepared.macro_indicator, lambda_value=config.lambda_value, slope_method=method)
        backtest = run_backtest(
            prepared,
            lambda_value=config.lambda_value,
            kappa=config.kappa,
            delta=config.delta,
            slope_method=method,
            oos_start=config.oos_start,
            oos_end=config.oos_end,
            benchmark_window=config.benchmark_window,
        )
        count_error = int(sum(abs(int(full_sample.counts[regime]) - PAPER_TABLE1_COUNTS[regime]) for regime in REGIME_ORDER))
        sharpe_delta = abs(float(backtest.summary.loc["Dynamic Sharpe", "Observed"]) - PAPER_TABLE5["Dynamic Sharpe"])
        rows.append(
            {
                "Slope Method": method,
                "Count Error": count_error,
                "Dynamic Sharpe": float(backtest.summary.loc["Dynamic Sharpe", "Observed"]),
                "Dynamic Sharpe Delta": sharpe_delta,
                "Dynamic Ann Return (%)": float(backtest.summary.loc["Dynamic Ann Return (%)", "Observed"]),
                "Information Ratio": float(backtest.summary.loc["Information Ratio", "Observed"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Count Error", "Dynamic Sharpe Delta"], ascending=[True, True]).reset_index(drop=True)


def run_parameter_sweep(
    prepared: PreparedData,
    base_config: KwonConfig,
    *,
    parameter_name: str,
    values: list[float],
) -> pd.DataFrame:
    rows = []
    for value in values:
        config = KwonConfig(
            data_end=base_config.data_end,
            fred_vintage_date=base_config.fred_vintage_date,
            lambda_value=value if parameter_name == "lambda_value" else base_config.lambda_value,
            kappa=value if parameter_name == "kappa" else base_config.kappa,
            delta=value if parameter_name == "delta" else base_config.delta,
            slope_method=base_config.slope_method,
            oos_start=base_config.oos_start,
            oos_end=base_config.oos_end,
            in_sample_end=base_config.in_sample_end,
            benchmark_window=base_config.benchmark_window,
        )
        backtest = run_backtest(
            prepared,
            lambda_value=config.lambda_value,
            kappa=config.kappa,
            delta=config.delta,
            slope_method=config.slope_method,
            oos_start=config.oos_start,
            oos_end=config.oos_end,
            benchmark_window=config.benchmark_window,
        )
        observed = backtest.summary["Observed"]
        rows.append(
            {
                "Parameter": parameter_name,
                "Value": value,
                "Dynamic Ann Return (%)": float(observed["Dynamic Ann Return (%)"]),
                "Dynamic Ann Vol (%)": float(observed["Dynamic Ann Vol (%)"]),
                "Dynamic Sharpe": float(observed["Dynamic Sharpe"]),
                "Dynamic Max DD (%)": float(observed["Dynamic Max DD (%)"]),
                "Dynamic Turnover (%)": float(observed["Dynamic Turnover (%)"]),
                "Tracking Error (%)": float(observed["Tracking Error (%)"]),
                "Information Ratio": float(observed["Information Ratio"]),
                "Break-even TC (%)": float(observed["Break-even TC (%)"]),
            }
        )
    return pd.DataFrame(rows)


def build_robustness_validation(frame: pd.DataFrame, parameter_name: str) -> pd.DataFrame:
    if parameter_name == "kappa":
        targets = PAPER_TABLE6
    elif parameter_name == "lambda_value":
        targets = PAPER_TABLE7
    elif parameter_name == "delta":
        targets = PAPER_TABLE8
    else:
        raise ValueError(f"Unsupported parameter for robustness validation: {parameter_name}")

    rows = []
    for _, row in frame.iterrows():
        target = targets.get(float(row["Value"]), {})
        for metric_name, paper_value in target.items():
            if metric_name == "Sharpe":
                observed_key = "Dynamic Sharpe"
            elif metric_name == "Ann Return (%)":
                observed_key = "Dynamic Ann Return (%)"
            else:
                observed_key = metric_name
            observed = float(row[observed_key])
            rows.append(
                {
                    "Parameter": parameter_name,
                    "Value": float(row["Value"]),
                    "Metric": metric_name,
                    "Paper": paper_value,
                    "Observed": observed,
                    "Abs Delta": abs(observed - paper_value),
                }
            )
    return pd.DataFrame(rows)


def save_dataframe(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
