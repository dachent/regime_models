#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_access import DEFAULT_CACHE_DIR, describe_series_ranges, load_raw_model_data
from model_core import ASSETS, REGIME_LABELS, build_production_outputs, save_production_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduction of Kim and Kwon (2023), Dynamic asset allocation strategy: an economic regime approach."
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Artifact output directory.")
    parser.add_argument("--lambda-value", type=float, default=0.3, help="l1 trend-filter penalty.")
    parser.add_argument("--delta", type=float, default=5.0, help="Black-Litterman risk-aversion parameter.")
    parser.add_argument("--kappa", type=float, default=0.09, help="Black-Litterman confidence parameter.")
    return parser.parse_args()


def print_console_summary(outputs: dict, notes: list[str]) -> None:
    table1 = outputs["table1"]
    table2 = outputs["table2"]
    table3 = outputs["table3"]
    summary = outputs["oos_summary"]
    bundle = outputs["bundle"]

    print("=" * 72)
    print("KIM AND KWON (2023) REGIME MODEL REPRODUCTION")
    print("=" * 72)
    for note in notes:
        print(f"- {note}")

    print("\nConfigured production parameters")
    print(f"  lambda={outputs['config']['lambda']:.2f}  delta={outputs['config']['delta']:.2f}  kappa={outputs['config']['kappa']:.2f}")
    print(f"  PCA explained variance: {bundle.explained_variance:.2%}")
    print("  PCA loadings:")
    for label, value in bundle.pca_loadings.sort_index().items():
        print(f"    {label:<18} {value:>8.4f}")

    print("\nTable 1 proxy comparison")
    print(table1[["Observed Return", "Paper Return", "Observed Vol", "Paper Vol", "Observed Sharpe", "Paper Sharpe"]].round(3).to_string())

    print("\nTable 2 sign matches")
    sign_matches = int(table2["Sign Match"].sum())
    print(f"  {sign_matches}/12 sign patterns match the paper.")
    print(table2[["Regime Label", "Asset", "Observed Return", "Paper Return", "Observed Sharpe", "Paper Sharpe", "Sign Match"]].round(3).to_string(index=False))

    print("\nTable 3 OOS comparison")
    print(table3.round(3).to_string(index=False))

    print("\nObserved OOS summary")
    print(summary.round(3).to_string())

    regime_counts = outputs["regimes"]["Regime"].value_counts().rename(index=REGIME_LABELS)
    print("\nRegime distribution, 1976-2020")
    print(regime_counts.to_string())


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    figures_dir = artifacts_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_model_data(cache_dir)
    outputs = build_production_outputs(raw_data, lam=args.lambda_value, delta=args.delta, kappa=args.kappa)

    print_console_summary(
        outputs,
        raw_data.notes
        + describe_series_ranges(
            [
                ("Growth indicator", outputs["bundle"].growth),
                ("Inflation indicator", outputs["bundle"].inflation),
                ("Risk-free series", raw_data.risk_free),
            ]
        ),
    )

    outputs["table1"].to_csv(artifacts_dir / "table1_proxy_comparison.csv")
    outputs["table2"].to_csv(artifacts_dir / "table2_regime_validation.csv", index=False)
    outputs["table3"].to_csv(artifacts_dir / "table3_oos_validation.csv", index=False)
    outputs["regimes"].to_csv(artifacts_dir / "regime_timeline.csv")
    outputs["oos"].to_csv(artifacts_dir / "oos_backtest.csv")
    outputs["oos_summary"].to_csv(artifacts_dir / "oos_summary.csv")
    outputs["bundle"].components_signed.to_csv(artifacts_dir / "growth_components_signed.csv")
    outputs["bundle"].components_z.to_csv(artifacts_dir / "growth_components_z.csv")

    metrics_payload = {
        "config": outputs["config"],
        "assets": ASSETS,
        "table3": outputs["table3"].to_dict(orient="records"),
        "pca_explained_variance": outputs["bundle"].explained_variance,
        "pca_loadings": outputs["bundle"].pca_loadings.to_dict(),
    }
    (artifacts_dir / "production_summary.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    save_production_figures(outputs, figures_dir)
    print(f"\nArtifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
