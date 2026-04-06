#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data_access import DEFAULT_CACHE_DIR, load_raw_model_data
from model_core import calibrate_kappa_for_tracking_error, run_lambda_scenarios, summarize_oos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded scenario tests for the Kim and Kwon (2023) reproduction.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Artifact output directory.")
    parser.add_argument("--lambdas", default="0.1,0.3,0.5", help="Comma-separated lambda values to test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    figures_dir = artifacts_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_model_data(cache_dir)
    lambdas = [float(item) for item in args.lambdas.split(",") if item.strip()]

    lambda_results = run_lambda_scenarios(raw_data, lambdas)
    lambda_results.to_csv(artifacts_dir / "lambda_scenarios.csv", index=False)

    calibrated_kappa, calibrated_oos = calibrate_kappa_for_tracking_error(raw_data)
    calibrated_summary = summarize_oos(calibrated_oos)
    calibrated_summary.to_csv(artifacts_dir / "kappa_calibration_summary.csv")
    calibrated_oos.to_csv(artifacts_dir / "kappa_calibration_backtest.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambda_results["Lambda"], lambda_results["Dynamic Sharpe"], marker="o", linewidth=2.0, label="Dynamic Sharpe")
    ax.plot(lambda_results["Lambda"], lambda_results["Tracking Error"], marker="o", linewidth=2.0, label="Tracking Error")
    ax.set_title("Lambda Sensitivity")
    ax.set_xlabel("Lambda")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "lambda_sensitivity.png", dpi=180)
    plt.close(fig)

    calibration_note = (
        f"Calibrated kappa={calibrated_kappa:.4f} achieves observed tracking error "
        f"{calibrated_summary.loc['Dynamic', 'Tracking Error']:.2f}%."
    )
    (artifacts_dir / "kappa_calibration_note.txt").write_text(calibration_note + "\n", encoding="utf-8")

    print("Lambda scenarios")
    print(lambda_results.round(3).to_string(index=False))
    print("\nKappa calibration")
    print(calibration_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
