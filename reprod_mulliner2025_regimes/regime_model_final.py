#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_access import DEFAULT_CACHE_DIR, load_production_data
from model_core import build_validation_summary, format_stats_for_console, run_regime_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime model reproduction for Mulliner et al. (2025) using aMDT-compatible data paths."
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory for fetched source data.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for generated CSV and chart artifacts.")
    parser.add_argument("--start", default="1985", help="Start year for reported performance.")
    parser.add_argument("--end", default="2024", help="End year for reported performance.")
    return parser.parse_args()


def print_results(stats: pd.DataFrame, performance_validation: pd.DataFrame, z_summary: pd.DataFrame, corr_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(format_stats_for_console(stats).to_string())

    print("\n" + "=" * 70)
    print("VALIDATION VS PAPER")
    print("=" * 70)
    for row in performance_validation.to_dict(orient="records"):
        print(
            f"  {row['Metric']:<22s} Paper: {row['Paper']:>5.2f}  "
            f"Observed: {row['Observed']:>5.2f}  Delta={row['Abs Delta']:.2f}  [{row['Status']}]"
        )

    print("\n  Z-score diagnostics")
    print(f"    {'Variable':<18s} {'Observed Mean':>14s} {'Paper Mean':>11s} {'Observed Std':>13s} {'Paper Std':>10s}")
    for row in z_summary.to_dict(orient="records"):
        print(
            f"    {row['Variable']:<18s} {row['Observed Mean']:>14.2f} {row['Paper Mean']:>11.2f} "
            f"{row['Observed Std']:>13.2f} {row['Paper Std']:>10.2f}"
        )

    print("\n  Cross-correlation checks")
    for row in corr_summary.to_dict(orient="records"):
        observed = row["Observed"]
        observed_text = f"{observed:+.2f}" if pd.notna(observed) else " n/a "
        print(
            f"    {row['Left']:>18s} x {row['Right']:<18s} Paper={row['Paper']:+.2f} "
            f"Observed={observed_text} [{row['Status']}]"
        )


def save_charts(results: pd.DataFrame, stats: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    cumulative = (1.0 + results[["Q1", "Q5", "LO", "Q1_Q5"]]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative.plot(ax=ax, linewidth=1.8)
    ax.set_title("Cumulative Returns, 1985-2024")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_returns.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Paper": {"Q1": 0.95, "Q5": 0.17, "LO": 1.00, "Q1_Q5": 0.82},
            "Observed": {
                "Q1": float(stats.loc["Q1", "Sharpe"]),
                "Q5": float(stats.loc["Q5", "Sharpe"]),
                "LO": float(stats.loc["LO", "Sharpe"]),
                "Q1_Q5": float(stats.loc["Q1_Q5", "Sharpe"]),
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


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    figures_dir = artifacts_dir / "figures"
    start_date = f"{args.start}-01-01"
    end_date = f"{args.end}-12-31"

    print("=" * 70)
    print("REGIME MODEL REPRODUCTION")
    print("Mulliner, Harvey, Xia, Fang and Van Hemert (2025)")
    print("=" * 70)

    econ_variables, factors, notes = load_production_data(cache_dir=cache_dir)
    print("\nLoaded data")
    for line in notes:
        print(f"  {line}")

    results, stats, aligned, corr_matrix = run_regime_model(
        econ_variables,
        factors,
        start_date=start_date,
        end_date=end_date,
    )
    performance_validation, z_summary, corr_summary = build_validation_summary(stats, corr_matrix, aligned)
    print_results(stats, performance_validation, z_summary, corr_summary)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "regime_results.csv")
    stats.to_csv(artifacts_dir / "performance_stats.csv")
    performance_validation.to_csv(artifacts_dir / "validation_summary.csv", index=False)
    z_summary.to_csv(artifacts_dir / "zscore_summary.csv", index=False)
    corr_summary.to_csv(artifacts_dir / "correlation_summary.csv", index=False)
    save_charts(results, stats, figures_dir)

    summary_payload = {
        "reporting_window": {"start": start_date, "end": end_date},
        "metrics": performance_validation.to_dict(orient="records"),
        "data_ranges": notes,
    }
    (artifacts_dir / "validation_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nArtifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
