from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from data_access import RawModelData


ASSETS = ["Stocks", "Bonds", "Commodities"]
REGIME_LABELS = {
    "HU": "Heating Up",
    "GL": "Goldilocks",
    "SG": "Slow Growth",
    "SF": "Stagflation",
}
REGIME_COLORS = {
    "HU": "#f97316",
    "GL": "#22c55e",
    "SG": "#3b82f6",
    "SF": "#ef4444",
}
TABLE1_PAPER = {
    "Stocks": {"Paper Return": 9.97, "Paper Vol": 15.03, "Paper Sharpe": 0.38},
    "Bonds": {"Paper Return": 7.03, "Paper Vol": 5.17, "Paper Sharpe": 0.53},
    "Commodities": {"Paper Return": 3.66, "Paper Vol": 20.17, "Paper Sharpe": -0.03},
}
TABLE2_PAPER = {
    ("HU", "Stocks"): {"Paper Return": 16.77, "Paper Vol": 11.07, "Paper Sharpe": 1.27},
    ("HU", "Bonds"): {"Paper Return": 4.32, "Paper Vol": 3.54, "Paper Sharpe": 0.47},
    ("HU", "Commodities"): {"Paper Return": 15.41, "Paper Vol": 17.86, "Paper Sharpe": 0.71},
    ("GL", "Stocks"): {"Paper Return": 17.19, "Paper Vol": 12.96, "Paper Sharpe": 0.99},
    ("GL", "Bonds"): {"Paper Return": 5.31, "Paper Vol": 4.63, "Paper Sharpe": 0.19},
    ("GL", "Commodities"): {"Paper Return": 0.12, "Paper Vol": 16.44, "Paper Sharpe": -0.26},
    ("SG", "Stocks"): {"Paper Return": 8.18, "Paper Vol": 17.21, "Paper Sharpe": 0.12},
    ("SG", "Bonds"): {"Paper Return": 14.02, "Paper Vol": 6.44, "Paper Sharpe": 1.24},
    ("SG", "Commodities"): {"Paper Return": -14.76, "Paper Vol": 20.98, "Paper Sharpe": -0.99},
    ("SF", "Stocks"): {"Paper Return": -3.27, "Paper Vol": 18.29, "Paper Sharpe": -0.45},
    ("SF", "Bonds"): {"Paper Return": 7.05, "Paper Vol": 5.94, "Paper Sharpe": 0.35},
    ("SF", "Commodities"): {"Paper Return": 5.03, "Paper Vol": 24.15, "Paper Sharpe": 0.00},
}
TABLE3_PAPER = {
    "Benchmark Return": 3.59,
    "Dynamic Return": 5.07,
    "Benchmark Vol": 6.46,
    "Dynamic Vol": 6.60,
    "Benchmark Sharpe": 0.56,
    "Dynamic Sharpe": 0.77,
    "Benchmark MaxDD": 28.99,
    "Dynamic MaxDD": 27.40,
    "Dynamic Turnover": 5.96,
    "Dynamic Tracking Error": 2.00,
    "Dynamic Information Ratio": 0.74,
    "Dynamic BreakEvenTC": 1.95,
}


@dataclass(frozen=True)
class IndicatorBundle:
    components_signed: pd.DataFrame
    components_z: pd.DataFrame
    growth: pd.Series
    inflation: pd.Series
    pca_loadings: pd.Series
    explained_variance: float


def _orient_series(primary: pd.Series, reference: pd.Series) -> pd.Series:
    aligned = pd.concat([primary, reference], axis=1).dropna()
    if aligned.empty:
        return primary
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.notna(corr) and corr < 0:
        return -primary
    return primary


def _scale_cpi_to_uig(cpi_yoy: pd.Series, uig: pd.Series) -> pd.Series:
    overlap = pd.concat([cpi_yoy, uig], axis=1, join="inner").dropna()
    if overlap.empty:
        raise RuntimeError("No CPI/UIG overlap available for inflation scaling.")

    cpi_overlap = overlap.iloc[:, 0]
    uig_overlap = overlap.iloc[:, 1]
    scaled = ((cpi_yoy - cpi_overlap.mean()) / cpi_overlap.std(ddof=0)) * uig_overlap.std(ddof=0) + uig_overlap.mean()
    scaled.name = "ScaledCPI"
    return scaled


def build_indicator_bundle(raw_data: RawModelData, *, end_date: str | pd.Timestamp) -> IndicatorBundle:
    end_ts = pd.Timestamp(end_date)

    growth_raw = raw_data.growth_raw.loc[:end_ts].copy()
    growth_raw["yield_spread"] = -growth_raw["yield_spread"]
    growth_raw["credit_spread"] = -growth_raw["credit_spread"]
    growth_raw["jobless_claims"] = -growth_raw["jobless_claims"].shift(1)
    growth_raw["building_permits"] = growth_raw["building_permits"].shift(1)
    growth_raw["volatility_proxy"] = -growth_raw["volatility_proxy"]
    growth_raw = growth_raw.dropna()

    components_z = (growth_raw - growth_raw.mean()) / growth_raw.std(ddof=0)
    components_z = components_z.dropna()
    if components_z.empty:
        raise RuntimeError("Growth components are empty after normalization.")

    pca = PCA(n_components=1)
    scores = pd.Series(
        pca.fit_transform(components_z).ravel(),
        index=components_z.index,
        name="Growth",
    )
    scores = _orient_series(scores, components_z.mean(axis=1))

    loadings = pd.Series(pca.components_[0], index=components_z.columns, name="Loading")
    if scores.corr(components_z.mean(axis=1)) < 0:
        loadings = -loadings

    cpi = raw_data.cpi.loc[:end_ts]
    cpi_yoy = cpi.pct_change(12) * 100.0
    cpi_yoy.name = "CPIYoY"
    uig = raw_data.uig.loc[:end_ts]
    scaled_cpi = _scale_cpi_to_uig(cpi_yoy, uig)
    inflation = scaled_cpi.where(scaled_cpi.index < uig.index.min(), uig.combine_first(scaled_cpi))
    inflation = inflation.shift(1).dropna()
    inflation.name = "Inflation"

    return IndicatorBundle(
        components_signed=growth_raw,
        components_z=components_z,
        growth=scores.dropna(),
        inflation=inflation.dropna(),
        pca_loadings=loadings,
        explained_variance=float(pca.explained_variance_ratio_[0]),
    )


def l1_trend_filter(series: pd.Series, lam: float = 0.3) -> pd.Series:
    y = series.dropna().astype(float)
    if len(y) < 3:
        return y.copy()

    n_obs = len(y)
    second_diff = sparse.diags(
        diagonals=[np.ones(n_obs - 2), -2.0 * np.ones(n_obs - 2), np.ones(n_obs - 2)],
        offsets=[0, 1, 2],
        shape=(n_obs - 2, n_obs),
        format="csr",
    )
    x = cp.Variable(n_obs)
    objective = cp.Minimize(0.5 * cp.sum_squares(x - y.values) + lam * cp.norm1(second_diff @ x))
    problem = cp.Problem(objective)

    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError:
            continue
        if x.value is not None:
            return pd.Series(np.asarray(x.value).ravel(), index=y.index, name=series.name)

    raise RuntimeError("L1 trend filter failed to converge.")


def classify_regimes(growth: pd.Series, inflation: pd.Series, *, lam: float = 0.3) -> pd.DataFrame:
    aligned = pd.concat([growth.rename("Growth"), inflation.rename("Inflation")], axis=1, join="inner").dropna()
    growth_trend = l1_trend_filter(aligned["Growth"], lam=lam).rename("GrowthTrend")
    inflation_trend = l1_trend_filter(aligned["Inflation"], lam=lam).rename("InflationTrend")

    growth_momentum = np.sign(growth_trend.diff()).replace(0.0, np.nan).ffill().bfill()
    inflation_momentum = np.sign(inflation_trend.diff()).replace(0.0, np.nan).ffill().bfill()

    regime = pd.Series(index=aligned.index, dtype="object", name="Regime")
    regime[(growth_momentum > 0) & (inflation_momentum > 0)] = "HU"
    regime[(growth_momentum > 0) & (inflation_momentum < 0)] = "GL"
    regime[(growth_momentum < 0) & (inflation_momentum < 0)] = "SG"
    regime[(growth_momentum < 0) & (inflation_momentum > 0)] = "SF"

    result = aligned.copy()
    result["GrowthTrend"] = growth_trend
    result["InflationTrend"] = inflation_trend
    result["GrowthMomentum"] = growth_momentum
    result["InflationMomentum"] = inflation_momentum
    result["Regime"] = regime.ffill().bfill()
    return result


def _annualize_return(monthly_returns: pd.Series) -> float:
    return float(monthly_returns.mean() * 12.0 * 100.0)


def _annualize_vol(monthly_returns: pd.Series) -> float:
    return float(monthly_returns.std(ddof=0) * np.sqrt(12.0) * 100.0)


def _sharpe(monthly_returns: pd.Series, risk_free: pd.Series) -> float:
    aligned = pd.concat([monthly_returns, risk_free], axis=1, join="inner").dropna()
    if aligned.empty:
        return float("nan")
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    ann_excess = excess.mean() * 12.0
    ann_vol = aligned.iloc[:, 0].std(ddof=0) * np.sqrt(12.0)
    if ann_vol == 0:
        return float("nan")
    return float(ann_excess / ann_vol)


def max_drawdown(monthly_returns: pd.Series) -> float:
    wealth = (1.0 + monthly_returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def performance_summary(monthly_returns: pd.Series, risk_free: pd.Series, *, label: str) -> dict[str, float | str]:
    aligned = pd.concat([monthly_returns.rename("ret"), risk_free.rename("rf")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise RuntimeError(f"No aligned return history for {label}.")

    active = aligned["ret"] - aligned["rf"]
    ann_return = aligned["ret"].mean() * 12.0
    ann_vol = aligned["ret"].std(ddof=0) * np.sqrt(12.0)
    sharpe = active.mean() * 12.0 / ann_vol if ann_vol else float("nan")
    summary = {
        "Label": label,
        "Ann Return": ann_return * 100.0,
        "Ann Vol": ann_vol * 100.0,
        "Sharpe": sharpe,
        "Max Drawdown": abs(max_drawdown(aligned["ret"])) * 100.0,
    }
    return summary


def compute_table1(asset_returns: pd.DataFrame, risk_free: pd.Series, *, start: str = "1976-01-31", end: str = "2020-12-31") -> pd.DataFrame:
    window = asset_returns.loc[start:end, ASSETS]
    rf_window = risk_free.loc[start:end]
    rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        returns = window[asset].dropna()
        rows.append(
            {
                "Asset": asset,
                "Observed Return": _annualize_return(returns),
                "Observed Vol": _annualize_vol(returns),
                "Observed Sharpe": _sharpe(returns, rf_window),
                **TABLE1_PAPER[asset],
            }
        )
    table = pd.DataFrame(rows).set_index("Asset")
    table["Return Delta"] = table["Observed Return"] - table["Paper Return"]
    table["Vol Delta"] = table["Observed Vol"] - table["Paper Vol"]
    table["Sharpe Delta"] = table["Observed Sharpe"] - table["Paper Sharpe"]
    return table


def compute_table2(asset_returns: pd.DataFrame, risk_free: pd.Series, regimes: pd.DataFrame, *, start: str = "1976-01-31", end: str = "2020-12-31") -> pd.DataFrame:
    frame = pd.concat([asset_returns[ASSETS], risk_free.rename("RiskFree"), regimes["Regime"]], axis=1, join="inner").loc[start:end]
    rows: list[dict[str, Any]] = []

    for regime_code, regime_name in REGIME_LABELS.items():
        regime_frame = frame[frame["Regime"] == regime_code]
        for asset in ASSETS:
            returns = regime_frame[asset].dropna()
            paper = TABLE2_PAPER[(regime_code, asset)]
            observed_sharpe = _sharpe(returns, regime_frame["RiskFree"])
            rows.append(
                {
                    "Regime": regime_code,
                    "Regime Label": regime_name,
                    "Asset": asset,
                    "Months": int(len(returns)),
                    "Observed Return": _annualize_return(returns),
                    "Observed Vol": _annualize_vol(returns),
                    "Observed Sharpe": observed_sharpe,
                    **paper,
                    "Return Delta": _annualize_return(returns) - paper["Paper Return"],
                    "Vol Delta": _annualize_vol(returns) - paper["Paper Vol"],
                    "Sharpe Delta": observed_sharpe - paper["Paper Sharpe"],
                    "Sign Match": np.sign(_annualize_return(returns)) == np.sign(paper["Paper Return"]) or abs(paper["Paper Return"]) < 1e-9,
                }
            )

    return pd.DataFrame(rows)


def fit_ar1_forecast(series: pd.Series) -> float:
    y = series.dropna().astype(float)
    if len(y) < 3:
        return float(y.iloc[-1])

    x = y.iloc[:-1].to_numpy()
    target = y.iloc[1:].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    forecast = beta[0] + beta[1] * y.iloc[-1]
    return float(forecast)


def forecast_next_regime(growth: pd.Series, inflation: pd.Series, *, lam: float = 0.3) -> tuple[str, pd.Series, pd.Series]:
    next_index = growth.index[-1] + pd.offsets.MonthEnd(1)
    growth_forecast = pd.concat([growth, pd.Series([fit_ar1_forecast(growth)], index=[next_index])])
    inflation_forecast = pd.concat([inflation, pd.Series([fit_ar1_forecast(inflation)], index=[next_index])])
    forecast_frame = classify_regimes(growth_forecast, inflation_forecast, lam=lam)
    regime = str(forecast_frame.iloc[-1]["Regime"])
    return regime, growth_forecast, inflation_forecast


def risk_parity_weights(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    n_assets = covariance.shape[0]

    def objective(weights: np.ndarray) -> float:
        port_vol = np.sqrt(weights @ covariance @ weights)
        if port_vol <= 0:
            return 1e9
        marginal = covariance @ weights / port_vol
        contributions = weights * marginal
        target = np.repeat(port_vol / n_assets, n_assets)
        return float(np.sum((contributions - target) ** 2))

    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    start = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return np.asarray(result.x, dtype=float)

    inv_vol = 1.0 / np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    return inv_vol / inv_vol.sum()


def mean_variance_weights(expected_returns: np.ndarray, covariance: np.ndarray, *, delta: float = 5.0) -> np.ndarray:
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(covariance.shape[0]) * 1e-8

    weights = cp.Variable(len(expected_returns), nonneg=True)
    objective = cp.Maximize(expected_returns @ weights - (delta / 2.0) * cp.quad_form(weights, covariance))
    problem = cp.Problem(objective, [cp.sum(weights) == 1.0])

    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError:
            continue
        if weights.value is not None:
            return np.asarray(weights.value).ravel()

    raise RuntimeError("Mean-variance optimization failed.")


def _post_return_weights(target_weights: np.ndarray, asset_returns: np.ndarray, portfolio_return: float) -> np.ndarray:
    gross = target_weights * (1.0 + asset_returns)
    denom = 1.0 + portfolio_return
    if denom <= 0:
        return target_weights.copy()
    return gross / denom


def _turnover_from_current(current_weights: np.ndarray | None, target_weights: np.ndarray) -> float:
    if current_weights is None:
        return 0.0
    return float(0.5 * np.abs(target_weights - current_weights).sum())


def break_even_transaction_cost(dynamic_returns: pd.Series, benchmark_returns: pd.Series, turnover: pd.Series) -> float:
    ann_dynamic = dynamic_returns.mean() * 12.0
    ann_benchmark = benchmark_returns.mean() * 12.0
    annual_turnover = turnover.mean() * 12.0
    if annual_turnover == 0:
        return float("nan")
    return float((ann_dynamic - ann_benchmark) / annual_turnover * 100.0)


def run_oos_backtest(
    raw_data: RawModelData,
    *,
    start: str = "2006-01-31",
    end: str = "2020-12-31",
    lookback_months: int = 360,
    lam: float = 0.3,
    delta: float = 5.0,
    kappa: float = 0.09,
) -> pd.DataFrame:
    asset_returns = raw_data.asset_returns[ASSETS].copy()
    risk_free = raw_data.risk_free.copy()
    months = asset_returns.loc[start:end].index

    rows: list[dict[str, Any]] = []
    current_benchmark_weights: np.ndarray | None = None
    current_dynamic_weights: np.ndarray | None = None

    for month in months:
        history_end = month - pd.offsets.MonthEnd(1)
        history_start = history_end - pd.offsets.MonthEnd(lookback_months - 1)
        window_returns = asset_returns.loc[history_start:history_end, ASSETS].dropna()
        if len(window_returns) < lookback_months:
            continue

        bundle = build_indicator_bundle(raw_data, end_date=history_end)
        historical_regimes = classify_regimes(bundle.growth, bundle.inflation, lam=lam).loc[history_start:history_end]
        forecast_regime, growth_forecast, inflation_forecast = forecast_next_regime(bundle.growth, bundle.inflation, lam=lam)

        covariance = window_returns.cov(ddof=0).to_numpy()
        benchmark_weights = risk_parity_weights(covariance)

        regime_join = window_returns.join(historical_regimes["Regime"], how="inner")
        forecast_subset = regime_join[regime_join["Regime"] == forecast_regime]
        if len(forecast_subset) < 6:
            q_view = window_returns.mean().to_numpy()
        else:
            q_view = forecast_subset[ASSETS].mean().to_numpy()

        implied_returns = delta * covariance @ benchmark_weights
        posterior_returns = (1.0 - kappa) * implied_returns + kappa * q_view
        dynamic_weights = mean_variance_weights(posterior_returns, covariance, delta=delta)

        realized_returns = asset_returns.loc[month, ASSETS].to_numpy(dtype=float)
        benchmark_return = float(benchmark_weights @ realized_returns)
        dynamic_return = float(dynamic_weights @ realized_returns)

        benchmark_turnover = _turnover_from_current(current_benchmark_weights, benchmark_weights)
        dynamic_turnover = _turnover_from_current(current_dynamic_weights, dynamic_weights)

        rows.append(
            {
                "Date": month,
                "Forecast Regime": forecast_regime,
                "Benchmark Return": benchmark_return,
                "Dynamic Return": dynamic_return,
                "Risk Free": float(risk_free.loc[month]),
                "Benchmark Turnover": benchmark_turnover,
                "Dynamic Turnover": dynamic_turnover,
                "Growth Forecast": float(growth_forecast.iloc[-1]),
                "Inflation Forecast": float(inflation_forecast.iloc[-1]),
                "Benchmark Stocks": benchmark_weights[0],
                "Benchmark Bonds": benchmark_weights[1],
                "Benchmark Commodities": benchmark_weights[2],
                "Dynamic Stocks": dynamic_weights[0],
                "Dynamic Bonds": dynamic_weights[1],
                "Dynamic Commodities": dynamic_weights[2],
            }
        )

        current_benchmark_weights = _post_return_weights(benchmark_weights, realized_returns, benchmark_return)
        current_dynamic_weights = _post_return_weights(dynamic_weights, realized_returns, dynamic_return)

    result = pd.DataFrame(rows).set_index("Date")
    if result.empty:
        raise RuntimeError("OOS backtest produced no rows.")
    return result


def summarize_oos(oos: pd.DataFrame) -> pd.DataFrame:
    dynamic = pd.Series(oos["Dynamic Return"], index=oos.index, name="Dynamic")
    benchmark = pd.Series(oos["Benchmark Return"], index=oos.index, name="Benchmark")
    risk_free = pd.Series(oos["Risk Free"], index=oos.index, name="RiskFree")

    dynamic_summary = performance_summary(dynamic, risk_free, label="Dynamic")
    benchmark_summary = performance_summary(benchmark, risk_free, label="Benchmark")
    active = dynamic - benchmark
    tracking_error = active.std(ddof=0) * np.sqrt(12.0) * 100.0
    information_ratio = (active.mean() * 12.0) / (tracking_error / 100.0) if tracking_error else float("nan")
    break_even = break_even_transaction_cost(dynamic, benchmark, oos["Dynamic Turnover"])

    table = pd.DataFrame([benchmark_summary, dynamic_summary]).set_index("Label")
    table.loc["Dynamic", "Tracking Error"] = tracking_error
    table.loc["Dynamic", "Information Ratio"] = information_ratio
    table.loc["Dynamic", "Avg Turnover"] = oos["Dynamic Turnover"].mean() * 100.0
    table.loc["Dynamic", "BreakEvenTC"] = break_even
    table.loc["Benchmark", "Avg Turnover"] = oos["Benchmark Turnover"].mean() * 100.0
    return table


def compare_table3(summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Benchmark Return", summary.loc["Benchmark", "Ann Return"], TABLE3_PAPER["Benchmark Return"]),
        ("Dynamic Return", summary.loc["Dynamic", "Ann Return"], TABLE3_PAPER["Dynamic Return"]),
        ("Benchmark Vol", summary.loc["Benchmark", "Ann Vol"], TABLE3_PAPER["Benchmark Vol"]),
        ("Dynamic Vol", summary.loc["Dynamic", "Ann Vol"], TABLE3_PAPER["Dynamic Vol"]),
        ("Benchmark Sharpe", summary.loc["Benchmark", "Sharpe"], TABLE3_PAPER["Benchmark Sharpe"]),
        ("Dynamic Sharpe", summary.loc["Dynamic", "Sharpe"], TABLE3_PAPER["Dynamic Sharpe"]),
        ("Benchmark MaxDD", summary.loc["Benchmark", "Max Drawdown"], TABLE3_PAPER["Benchmark MaxDD"]),
        ("Dynamic MaxDD", summary.loc["Dynamic", "Max Drawdown"], TABLE3_PAPER["Dynamic MaxDD"]),
        ("Dynamic Turnover", summary.loc["Dynamic", "Avg Turnover"], TABLE3_PAPER["Dynamic Turnover"]),
        ("Dynamic Tracking Error", summary.loc["Dynamic", "Tracking Error"], TABLE3_PAPER["Dynamic Tracking Error"]),
        ("Dynamic Information Ratio", summary.loc["Dynamic", "Information Ratio"], TABLE3_PAPER["Dynamic Information Ratio"]),
        ("Dynamic BreakEvenTC", summary.loc["Dynamic", "BreakEvenTC"], TABLE3_PAPER["Dynamic BreakEvenTC"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Observed", "Paper"]).assign(Delta=lambda frame: frame["Observed"] - frame["Paper"])


def calibrate_kappa_for_tracking_error(
    raw_data: RawModelData,
    *,
    target_tracking_error: float = 2.0,
    lam: float = 0.3,
    delta: float = 5.0,
    baseline_kappa: float = 0.09,
    baseline_result: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    if baseline_result is None:
        baseline_result = run_oos_backtest(raw_data, lam=lam, delta=delta, kappa=baseline_kappa)

    baseline_summary = summarize_oos(baseline_result)
    baseline_te = float(baseline_summary.loc["Dynamic", "Tracking Error"])
    if baseline_te <= 0:
        raise RuntimeError("Baseline tracking error is non-positive.")

    candidate = max(0.0, min(0.5, baseline_kappa * target_tracking_error / baseline_te))
    calibrated_result = run_oos_backtest(raw_data, lam=lam, delta=delta, kappa=candidate)
    return candidate, calibrated_result


def run_lambda_scenarios(raw_data: RawModelData, lambdas: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lam in lambdas:
        bundle = build_indicator_bundle(raw_data, end_date="2020-12-31")
        regimes = classify_regimes(bundle.growth, bundle.inflation, lam=lam)
        table2 = compute_table2(raw_data.asset_returns, raw_data.risk_free, regimes)
        oos = run_oos_backtest(raw_data, lam=lam)
        summary = summarize_oos(oos)
        rows.append(
            {
                "Lambda": lam,
                "Dynamic Sharpe": summary.loc["Dynamic", "Sharpe"],
                "Benchmark Sharpe": summary.loc["Benchmark", "Sharpe"],
                "Dynamic Return": summary.loc["Dynamic", "Ann Return"],
                "Tracking Error": summary.loc["Dynamic", "Tracking Error"],
                "Dynamic Turnover": summary.loc["Dynamic", "Avg Turnover"],
                "Sign Matches": int(table2["Sign Match"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_production_outputs(raw_data: RawModelData, *, lam: float = 0.3, delta: float = 5.0, kappa: float = 0.09) -> dict[str, Any]:
    bundle = build_indicator_bundle(raw_data, end_date="2020-12-31")
    regimes = classify_regimes(bundle.growth, bundle.inflation, lam=lam)
    table1 = compute_table1(raw_data.asset_returns, raw_data.risk_free)
    table2 = compute_table2(raw_data.asset_returns, raw_data.risk_free, regimes)
    oos = run_oos_backtest(raw_data, lam=lam, delta=delta, kappa=kappa)
    oos_summary = summarize_oos(oos)
    table3 = compare_table3(oos_summary)

    return {
        "bundle": bundle,
        "regimes": regimes,
        "table1": table1,
        "table2": table2,
        "oos": oos,
        "oos_summary": oos_summary,
        "table3": table3,
        "config": {"lambda": lam, "delta": delta, "kappa": kappa},
    }


def save_production_figures(outputs: dict[str, Any], figures_dir: Any) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle: IndicatorBundle = outputs["bundle"]
    regimes = outputs["regimes"]
    oos = outputs["oos"]

    plt.style.use("default")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    aligned = regimes.index
    axes[0].plot(bundle.growth.reindex(aligned), color="#2563eb", linewidth=1.2, label="Growth")
    axes[0].plot(regimes["GrowthTrend"], color="#0f172a", linewidth=2.0, label="L1 Trend")
    axes[0].axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.8)
    axes[0].set_title("Growth Indicator (PCA PC1) and Trend")
    axes[0].legend(frameon=False)

    axes[1].plot(bundle.inflation.reindex(aligned), color="#ea580c", linewidth=1.2, label="Inflation")
    axes[1].plot(regimes["InflationTrend"], color="#7c2d12", linewidth=2.0, label="L1 Trend")
    axes[1].axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.8)
    axes[1].set_title("Inflation Indicator (UIG/CPI Splice) and Trend")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "indicator_trends.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    colors = [REGIME_COLORS[code] for code in regimes["Regime"]]
    ax.bar(regimes.index, np.ones(len(regimes)), width=25, color=colors)
    ax.set_title("Regime Timeline, 1976-2020")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_timeline.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    wealth = pd.DataFrame(
        {
            "Benchmark": (1.0 + oos["Benchmark Return"]).cumprod(),
            "Dynamic": (1.0 + oos["Dynamic Return"]).cumprod(),
        }
    )
    wealth.plot(ax=ax, linewidth=2.0)
    ax.set_title("OOS Cumulative Wealth, 2006-2020")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_wealth.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(oos.index, oos["Dynamic Stocks"] * 100.0, color="#2563eb", linewidth=1.4)
    axes[0].set_ylabel("Stocks %")
    axes[1].plot(oos.index, oos["Dynamic Bonds"] * 100.0, color="#16a34a", linewidth=1.4)
    axes[1].set_ylabel("Bonds %")
    axes[2].plot(oos.index, oos["Dynamic Commodities"] * 100.0, color="#f97316", linewidth=1.4)
    axes[2].set_ylabel("Comm %")
    axes[0].set_title("Dynamic Portfolio Weights, 2006-2020")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(figures_dir / "portfolio_weights.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    table1 = outputs["table1"]
    paper_vol = [table1.loc[asset, "Paper Vol"] for asset in ASSETS]
    observed_vol = [table1.loc[asset, "Observed Vol"] for asset in ASSETS]
    x = np.arange(len(ASSETS))
    width = 0.35
    ax.bar(x - width / 2, observed_vol, width=width, label="Observed proxy")
    ax.bar(x + width / 2, paper_vol, width=width, label="Paper")
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS)
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Proxy Volatility Gap vs Paper Table 1")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "proxy_volatility_gap.png", dpi=180)
    plt.close(fig)
