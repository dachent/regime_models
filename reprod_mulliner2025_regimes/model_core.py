from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def modified_duration(ytm_pct: float, maturity: int = 10, freq: int = 2) -> float:
    y = ytm_pct / 100.0
    if y <= 0.001:
        return float(maturity)

    coupon = y / freq
    periods = max(int(maturity * freq), 1)
    pv_sum = 0.0
    time_pv_sum = 0.0
    for period in range(1, periods + 1):
        cash_flow = coupon if period < periods else (1.0 + coupon)
        pv = cash_flow / (1.0 + coupon) ** period
        pv_sum += pv
        time_pv_sum += period * pv

    if pv_sum == 0:
        return float(maturity)

    macaulay_duration = time_pv_sum / (pv_sum * freq)
    return macaulay_duration / (1.0 + y / freq)


def compute_bond_total_returns(yields_daily: pd.Series, maturity: int = 10) -> pd.Series:
    yields_clean = yields_daily.dropna().sort_index()
    returns = pd.Series(index=yields_clean.index[1:], dtype=float)
    for idx in range(1, len(yields_clean)):
        previous_yield = yields_clean.iloc[idx - 1]
        current_yield = yields_clean.iloc[idx]
        if pd.isna(previous_yield) or pd.isna(current_yield) or previous_yield <= 0:
            returns.iloc[idx - 1] = 0.0
            continue

        days = max(1, min(10, (yields_clean.index[idx] - yields_clean.index[idx - 1]).days))
        carry = (previous_yield / 100.0) * (days / 365.0)
        duration = modified_duration(previous_yield, maturity=maturity)
        price_return = -duration * (current_yield - previous_yield) / 100.0
        returns.iloc[idx - 1] = carry + price_return

    return returns


def transform_variable(series: pd.Series, *, is_pct_change: bool = False) -> pd.Series:
    clean = series.dropna().sort_index()
    changes = clean.pct_change(12) if is_pct_change else clean.diff(12)
    changes = changes.dropna()
    rolling_std = changes.rolling(120, min_periods=60).std()
    z_scores = (changes / rolling_std).clip(-3, 3)
    return z_scores.dropna()


def compute_sb_correlation(
    stock_returns_daily: pd.Series,
    bond_returns_daily: pd.Series,
    *,
    window: int = 756,
    min_periods: int = 500,
) -> pd.Series:
    common_index = stock_returns_daily.index.intersection(bond_returns_daily.index)
    stock = stock_returns_daily.loc[common_index]
    bond = bond_returns_daily.loc[common_index]
    rolling_corr = stock.rolling(window, min_periods=min_periods).corr(bond)
    return rolling_corr.resample("ME").last().dropna()


def run_regime_model(
    econ_variables: dict[str, pd.Series],
    factor_returns: pd.DataFrame,
    *,
    exclude_months: int = 36,
    n_quintiles: int = 5,
    start_date: str = "1985-01-01",
    end_date: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pct_change_vars = {"Market", "Oil", "Copper"}
    transformed = {
        name: transform_variable(series, is_pct_change=name in pct_change_vars)
        for name, series in econ_variables.items()
    }

    aligned = pd.DataFrame(transformed).dropna()
    if aligned.empty:
        raise RuntimeError("No aligned monthly observations were produced for the economic variables.")

    corr_matrix = aligned.corr()
    values = aligned.values
    dates = aligned.index
    n_rows = len(dates)

    subsequent_factor_returns: dict[int, tuple[pd.Timestamp, pd.Series]] = {}
    for idx, current_date in enumerate(dates):
        future_dates = factor_returns.index[factor_returns.index > current_date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]
        next_returns = factor_returns.loc[next_date]
        if next_returns.isna().any():
            continue
        subsequent_factor_returns[idx] = (next_date, next_returns)

    quintile_returns: dict[int, list[float]] = {bucket: [] for bucket in range(1, n_quintiles + 1)}
    quintile_dates: list[pd.Timestamp] = []
    long_only_returns: list[float] = []

    for current_idx in range(exclude_months + 20, n_rows):
        if current_idx not in subsequent_factor_returns:
            continue
        next_date, next_returns = subsequent_factor_returns[current_idx]
        history_end = current_idx - exclude_months
        if history_end < 20:
            continue

        distances = np.sqrt(np.sum((values[:history_end] - values[current_idx]) ** 2, axis=1))
        valid_history = sorted(
            [(hist_idx, distances[hist_idx]) for hist_idx in range(history_end) if hist_idx in subsequent_factor_returns],
            key=lambda item: item[1],
        )
        if len(valid_history) < n_quintiles * 10:
            continue

        items_per_quintile = len(valid_history) // n_quintiles
        quintile_dates.append(next_date)
        for quintile in range(1, n_quintiles + 1):
            members = valid_history[(quintile - 1) * items_per_quintile : quintile * items_per_quintile]
            average_subsequent_return = pd.DataFrame(
                [subsequent_factor_returns[hist_idx][1] for hist_idx, _distance in members]
            ).mean()
            positions = np.sign(average_subsequent_return)
            quintile_returns[quintile].append(float((positions * next_returns).mean()))

        long_only_returns.append(float(next_returns.mean()))

    results = pd.DataFrame(
        {f"Q{bucket}": quintile_returns[bucket] for bucket in range(1, n_quintiles + 1)},
        index=quintile_dates,
    )
    results["LO"] = long_only_returns
    results["Q1_Q5"] = results["Q1"] - results["Q5"]

    filtered = results.loc[(results.index >= start_date) & (results.index <= end_date)]
    if filtered.empty:
        raise RuntimeError("The selected reporting window produced no strategy returns.")

    stats = compute_performance_stats(filtered)
    return filtered, stats, aligned, corr_matrix


def compute_performance_stats(results: pd.DataFrame) -> pd.DataFrame:
    stats: dict[str, dict[str, float | int]] = {}
    for column in results.columns:
        series = results[column].dropna()
        ann_return = float(series.mean() * 12.0)
        ann_vol = float(series.std() * np.sqrt(12.0))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cumulative = (1.0 + series).cumprod()
        max_drawdown = float(((cumulative / cumulative.cummax()) - 1.0).min())
        t_stat = float(series.mean() / (series.std() / np.sqrt(len(series)))) if series.std() > 0 else 0.0
        corr_lo = float(series.corr(results["LO"])) if column != "LO" else 1.0

        stats[column] = {
            "Ann Return": ann_return,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Max DD": max_drawdown,
            "Corr(LO)": corr_lo,
            "t-stat": t_stat,
            "N": int(len(series)),
        }

    return pd.DataFrame(stats).T


def build_validation_summary(
    stats: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_rows = [
        ("Q1 Sharpe", 0.95, float(stats.loc["Q1", "Sharpe"])),
        ("Q5 Sharpe", 0.17, float(stats.loc["Q5", "Sharpe"])),
        ("LO Sharpe", 1.00, float(stats.loc["LO", "Sharpe"])),
        ("Q1-Q5 Sharpe", 0.82, float(stats.loc["Q1_Q5", "Sharpe"])),
        ("Q1 Corr(LO)", 0.76, float(stats.loc["Q1", "Corr(LO)"])),
        ("Q5 Corr(LO)", 0.48, float(stats.loc["Q5", "Corr(LO)"])),
        ("Q1-Q5 t-stat", 3.00, float(stats.loc["Q1_Q5", "t-stat"])),
    ]
    performance_validation = pd.DataFrame(performance_rows, columns=["Metric", "Paper", "Observed"])
    performance_validation["Abs Delta"] = (performance_validation["Paper"] - performance_validation["Observed"]).abs()
    performance_validation["Status"] = performance_validation["Abs Delta"].apply(
        lambda delta: "MATCH" if delta < 0.10 else ("CLOSE" if delta < 0.25 else "GAP")
    )

    paper_z = {
        "Market": (0.49, 0.89),
        "Yield_curve": (-0.07, 0.98),
        "Oil": (0.24, 0.97),
        "Copper": (0.15, 0.94),
        "Monetary_policy": (0.13, 0.98),
        "Volatility": (0.05, 0.97),
        "Stock_bond": (-0.10, 0.98),
    }
    z_rows = []
    for variable in aligned.columns:
        paper_mean, paper_std = paper_z.get(variable, (np.nan, np.nan))
        z_rows.append(
            {
                "Variable": variable,
                "Observed Mean": float(aligned[variable].mean()),
                "Paper Mean": paper_mean,
                "Observed Std": float(aligned[variable].std()),
                "Paper Std": paper_std,
            }
        )
    z_summary = pd.DataFrame(z_rows)
    z_summary["Mean Delta"] = (z_summary["Observed Mean"] - z_summary["Paper Mean"]).abs()
    z_summary["Std Delta"] = (z_summary["Observed Std"] - z_summary["Paper Std"]).abs()

    correlation_checks = [
        ("Oil", "Copper", 0.33),
        ("Monetary_policy", "Yield_curve", -0.67),
        ("Volatility", "Market", -0.25),
        ("Monetary_policy", "Stock_bond", -0.36),
        ("Copper", "Monetary_policy", 0.37),
        ("Copper", "Market", 0.13),
    ]
    corr_rows = []
    for left, right, paper_value in correlation_checks:
        observed = np.nan
        if left in corr_matrix.index and right in corr_matrix.columns:
            observed = float(corr_matrix.loc[left, right])
        corr_rows.append(
            {
                "Left": left,
                "Right": right,
                "Paper": paper_value,
                "Observed": observed,
                "Abs Delta": abs(paper_value - observed) if pd.notna(observed) else np.nan,
            }
        )
    corr_summary = pd.DataFrame(corr_rows)
    corr_summary["Status"] = corr_summary["Abs Delta"].apply(
        lambda delta: "MATCH" if pd.notna(delta) and delta < 0.15 else ("CLOSE" if pd.notna(delta) and delta < 0.30 else "GAP")
    )

    return performance_validation, z_summary, corr_summary


def format_stats_for_console(stats: pd.DataFrame) -> pd.DataFrame:
    formatted = stats.copy()
    for column in ["Ann Return", "Ann Vol", "Max DD"]:
        formatted[column] = formatted[column].apply(lambda value: f"{value:.2%}")
    for column in ["Sharpe", "Corr(LO)", "t-stat"]:
        formatted[column] = formatted[column].apply(lambda value: f"{value:.2f}")
    formatted["N"] = formatted["N"].astype(int)
    return formatted


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


def to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.reset_index().rename(columns={"index": "Index"}).to_dict(orient="records")


def print_table(title: str, lines: Iterable[str]) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    for line in lines:
        print(line)
