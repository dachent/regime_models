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
    build_robustness_validation,
    build_slope_method_comparison,
    build_table1_validation,
    build_table2_validation,
    build_table3_validation,
    build_table4_validation,
    build_table5_validation,
    run_baseline_analysis,
    run_parameter_sweep,
    save_dataframe,
)
from regime_model_final import config_from_args, save_baseline_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full validation and robustness runner for Kwon (2022).")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory for fetched source data.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for generated CSV, JSON, and figure artifacts.")
    parser.add_argument("--data-end", default="2021-10-31", help="Last observation date to include in the validation dataset.")
    parser.add_argument("--fred-vintage-date", default="2021-10-31", help="FRED vintage date for revised macro series. Use 'none' for current vintage.")
    parser.add_argument("--lambda-value", type=float, default=0.5, help="Canonical L1 trend filtering penalty.")
    parser.add_argument("--kappa", type=float, default=0.5, help="Canonical Black-Litterman confidence parameter.")
    parser.add_argument("--delta", type=float, default=5.0, help="Canonical Black-Litterman risk aversion parameter.")
    parser.add_argument("--slope-method", choices=["forward", "backward", "center"], default="center", help="Canonical slope convention used for regime classification.")
    parser.add_argument("--benchmark-window", type=int, default=0, help="Optional rolling covariance window for diagnostics. Leave at 0 for canonical expanding window.")
    return parser.parse_args()


def save_robustness_chart(kappa_frame: pd.DataFrame, lambda_frame: pd.DataFrame, delta_frame: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for axis, frame, title in zip(
        axes,
        [kappa_frame, lambda_frame, delta_frame],
        ["Table 6: Kappa Sweep", "Table 7: Lambda Sweep", "Table 8: Delta Sweep"],
    ):
        axis.plot(frame["Value"], frame["Dynamic Sharpe"], marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Value")
        axis.set_ylabel("Dynamic Sharpe")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "robustness_sweeps.png", dpi=160)
    plt.close(fig)


def build_validation_payload(
    analysis,
    slope_comparison: pd.DataFrame,
    kappa_frame: pd.DataFrame,
    lambda_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "config": {
            "data_end": analysis.config.data_end,
            "fred_vintage_date": analysis.config.fred_vintage_date,
            "lambda_value": analysis.config.lambda_value,
            "kappa": analysis.config.kappa,
            "delta": analysis.config.delta,
            "slope_method": analysis.config.slope_method,
            "benchmark_window": analysis.config.benchmark_window,
        },
        "table5": build_table5_validation(analysis.backtest.summary).to_dict(orient="records"),
        "slope_method_comparison": slope_comparison.to_dict(orient="records"),
        "robustness": {
            "kappa": kappa_frame.to_dict(orient="records"),
            "lambda": lambda_frame.to_dict(orient="records"),
            "delta": delta_frame.to_dict(orient="records"),
        },
    }


def print_validation_summary(table5_validation: pd.DataFrame, slope_comparison: pd.DataFrame) -> None:
    print("=" * 72)
    print("KWON (2022) VALIDATION + ROBUSTNESS")
    print("=" * 72)
    print("Primary OOS deltas vs paper:")
    for row in table5_validation.to_dict(orient="records"):
        print(f"  {row['Metric']:<28s} paper={row['Paper']:>8.3f} observed={row['Observed']:>8.3f} delta={row['Abs Delta']:>8.3f}")
    print("\nSlope-method ranking:")
    print(slope_comparison.to_string(index=False))


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    figures_dir = artifacts_dir / "figures"
    config: KwonConfig = config_from_args(args)

    raw_data = load_kwon_raw_data(
        cache_dir=cache_dir,
        data_end=config.data_end,
        fred_vintage_date=config.fred_vintage_date,
    )
    analysis = run_baseline_analysis(raw_data, config)
    save_baseline_artifacts(analysis, artifacts_dir)

    table1_counts, table1_transitions = build_table1_validation(analysis.full_sample_regimes)
    table2_validation = build_table2_validation(analysis.factor_descriptive_stats)
    table3_validation = build_table3_validation(analysis.regime_factor_stats)
    table4_validation = build_table4_validation(analysis.in_sample_allocations)
    table5_validation = build_table5_validation(analysis.backtest.summary)
    slope_comparison = build_slope_method_comparison(analysis.prepared, config)

    kappa_frame = run_parameter_sweep(analysis.prepared, config, parameter_name="kappa", values=[0.3, 0.5, 0.7])
    lambda_frame = run_parameter_sweep(analysis.prepared, config, parameter_name="lambda_value", values=[0.3, 0.5, 0.7])
    delta_frame = run_parameter_sweep(analysis.prepared, config, parameter_name="delta", values=[3.0, 5.0, 7.0])

    robustness_validation = pd.concat(
        [
            build_robustness_validation(kappa_frame, "kappa"),
            build_robustness_validation(lambda_frame, "lambda_value"),
            build_robustness_validation(delta_frame, "delta"),
        ],
        ignore_index=True,
    )

    save_dataframe(table1_counts, artifacts_dir / "table1_regime_counts_validation.csv", index=False)
    save_dataframe(table1_transitions, artifacts_dir / "table1_transition_validation.csv", index=False)
    save_dataframe(table2_validation, artifacts_dir / "table2_validation.csv", index=False)
    save_dataframe(table3_validation, artifacts_dir / "table3_validation.csv", index=False)
    save_dataframe(table4_validation, artifacts_dir / "table4_validation.csv", index=False)
    save_dataframe(table5_validation, artifacts_dir / "table5_validation.csv", index=False)
    save_dataframe(slope_comparison, artifacts_dir / "slope_method_comparison.csv", index=False)
    save_dataframe(kappa_frame, artifacts_dir / "robustness_kappa.csv", index=False)
    save_dataframe(lambda_frame, artifacts_dir / "robustness_lambda.csv", index=False)
    save_dataframe(delta_frame, artifacts_dir / "robustness_delta.csv", index=False)
    save_dataframe(robustness_validation, artifacts_dir / "robustness_validation.csv", index=False)

    save_robustness_chart(kappa_frame, lambda_frame, delta_frame, figures_dir)

    payload = build_validation_payload(analysis, slope_comparison, kappa_frame, lambda_frame, delta_frame)
    (artifacts_dir / "validation_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_validation_summary(table5_validation, slope_comparison)
    print(f"\nValidation artifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
