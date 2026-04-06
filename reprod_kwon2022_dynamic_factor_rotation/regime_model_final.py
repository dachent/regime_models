#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_access import DEFAULT_CACHE_DIR, load_kwon_raw_data
from model_core import (
    KwonConfig,
    PAPER_TABLE5,
    REGIME_ORDER,
    build_table5_validation,
    run_baseline_analysis,
    save_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical reproduction runner for Kwon (2022).")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory for fetched source data.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for generated CSV, JSON, and figure artifacts.")
    parser.add_argument("--data-end", default="2021-10-31", help="Last observation date to include in the reproduction dataset.")
    parser.add_argument("--fred-vintage-date", default="2021-10-31", help="FRED vintage date for revised macro series. Use 'none' for current vintage.")
    parser.add_argument("--lambda-value", type=float, default=0.5, help="L1 trend filtering penalty.")
    parser.add_argument("--kappa", type=float, default=0.5, help="Black-Litterman confidence parameter.")
    parser.add_argument("--delta", type=float, default=5.0, help="Black-Litterman risk aversion parameter.")
    parser.add_argument("--slope-method", choices=["forward", "backward", "center"], default="center", help="Slope convention used for regime classification.")
    parser.add_argument("--benchmark-window", type=int, default=0, help="Optional rolling covariance window. Leave at 0 for the canonical expanding-window benchmark.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> KwonConfig:
    return KwonConfig(
        data_end=args.data_end,
        fred_vintage_date=None if str(args.fred_vintage_date).lower() == "none" else args.fred_vintage_date,
        lambda_value=float(args.lambda_value),
        kappa=float(args.kappa),
        delta=float(args.delta),
        slope_method=args.slope_method,
        benchmark_window=(None if int(args.benchmark_window) <= 0 else int(args.benchmark_window)),
    )


def build_macro_frame(analysis) -> pd.DataFrame:
    return pd.concat(
        [
            analysis.prepared.macro_indicator.rename("macro_indicator"),
            analysis.full_sample_regimes.trend.rename("l1_trend"),
            analysis.full_sample_regimes.slope.rename("slope"),
            analysis.full_sample_regimes.regimes.rename("regime"),
        ],
        axis=1,
    )


def build_oos_return_frame(analysis) -> pd.DataFrame:
    return analysis.backtest.returns.join(analysis.backtest.forecast_regimes)


def save_baseline_figures(analysis, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    cumulative = (1.0 + analysis.backtest.returns[["Benchmark", "Dynamic"]]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative.plot(ax=ax, linewidth=1.8)
    ax.set_title("Kwon (2022) OOS Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_returns.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Paper": {
                "Benchmark Sharpe": PAPER_TABLE5["Benchmark Sharpe"],
                "Dynamic Sharpe": PAPER_TABLE5["Dynamic Sharpe"],
            },
            "Observed": {
                "Benchmark Sharpe": float(analysis.backtest.summary.loc["Benchmark Sharpe", "Observed"]),
                "Dynamic Sharpe": float(analysis.backtest.summary.loc["Dynamic Sharpe", "Observed"]),
            },
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Sharpe Ratio Comparison")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "sharpe_comparison.png", dpi=160)
    plt.close(fig)

    counts = analysis.full_sample_regimes.counts.reindex(REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#2196F3", "#FF9800", "#F44336"])
    ax.set_title("Full-Sample Regime Counts")
    ax.set_ylabel("Months")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "regime_counts.png", dpi=160)
    plt.close(fig)


def save_baseline_artifacts(analysis, artifacts_dir: Path) -> None:
    figures_dir = artifacts_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(build_macro_frame(analysis), artifacts_dir / "macro_indicator.csv")
    save_dataframe(analysis.prepared.raw_macro, artifacts_dir / "macro_raw_variables.csv")
    save_dataframe(analysis.prepared.pc1_loadings.to_frame("pc1_loading"), artifacts_dir / "pca_loadings.csv")
    save_dataframe(analysis.prepared.factors, artifacts_dir / "factor_returns.csv")
    save_dataframe(analysis.factor_descriptive_stats, artifacts_dir / "factor_descriptive_stats.csv")
    save_dataframe(analysis.factor_correlation_matrix, artifacts_dir / "factor_correlation_matrix.csv")
    save_dataframe(analysis.regime_factor_stats, artifacts_dir / "regime_factor_stats.csv", index=False)
    save_dataframe(analysis.in_sample_allocations, artifacts_dir / "in_sample_allocations.csv")
    save_dataframe(build_oos_return_frame(analysis), artifacts_dir / "oos_returns.csv")
    save_dataframe(analysis.backtest.weights_rp, artifacts_dir / "weights_risk_parity.csv")
    save_dataframe(analysis.backtest.weights_dynamic, artifacts_dir / "weights_dynamic.csv")
    save_dataframe(analysis.backtest.summary, artifacts_dir / "oos_summary.csv")
    save_dataframe(build_table5_validation(analysis.backtest.summary), artifacts_dir / "table5_validation.csv", index=False)

    summary_payload = {
        "paper": "Kwon (2022) Dynamic Factor Rotation Strategy: A Business Cycle Approach",
        "config": {
            "data_end": analysis.config.data_end,
            "fred_vintage_date": analysis.config.fred_vintage_date,
            "lambda_value": analysis.config.lambda_value,
            "kappa": analysis.config.kappa,
            "delta": analysis.config.delta,
            "slope_method": analysis.config.slope_method,
            "benchmark_window": analysis.config.benchmark_window,
        },
        "pc1_explained_variance_ratio": analysis.prepared.pc1_explained_variance_ratio,
        "table5": build_table5_validation(analysis.backtest.summary).to_dict(orient="records"),
    }
    (artifacts_dir / "baseline_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    save_baseline_figures(analysis, figures_dir)


def print_console_summary(analysis) -> None:
    summary = analysis.backtest.summary["Observed"]
    print("=" * 72)
    print("KWON (2022) CANONICAL REPRODUCTION")
    print("=" * 72)
    print(f"Macro PC1 explained variance: {analysis.prepared.pc1_explained_variance_ratio * 100.0:.2f}%")
    print(f"Full-sample regime counts: {analysis.full_sample_regimes.counts.to_dict()}")
    print("")
    for metric in [
        "Benchmark Ann Return (%)",
        "Benchmark Sharpe",
        "Dynamic Ann Return (%)",
        "Dynamic Sharpe",
        "Information Ratio",
        "Dynamic Turnover (%)",
    ]:
        print(f"{metric:<28s} {float(summary[metric]):>10.3f}")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    config = config_from_args(args)
    raw_data = load_kwon_raw_data(
        cache_dir=cache_dir,
        data_end=config.data_end,
        fred_vintage_date=config.fred_vintage_date,
    )
    analysis = run_baseline_analysis(raw_data, config)
    save_baseline_artifacts(analysis, artifacts_dir)
    print_console_summary(analysis)
    print(f"\nArtifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
