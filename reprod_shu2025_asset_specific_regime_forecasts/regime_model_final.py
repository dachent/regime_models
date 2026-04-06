#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from data_access import DEFAULT_CACHE_DIR, load_all_data
from model_core import (
    ASSETS,
    DATA_END,
    DATA_START,
    DEFAULT_WORKERS,
    PROXY_FUNDS,
    format_stats_for_console,
    run_model,
    save_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime model reproduction for Shu, Yu, and Mulvey (2025) using aMDT-compatible data paths."
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory for fetched source data.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for generated CSV and chart artifacts.")
    parser.add_argument("--start", default=DATA_START, help="Fetch start date.")
    parser.add_argument("--end", default=DATA_END, help="Fetch end date.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Maximum concurrent asset workers for rolling lambda selection.")
    parser.add_argument("--lambda-seed", default="", help="Optional pickle or CSV file containing precomputed rolling lambdas.")
    return parser.parse_args()


def load_lambda_seed(path: Path) -> dict[str, object]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        required = {"Asset", "Refit Date", "Best Lambda"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"CSV seed file must contain columns {sorted(required)}.")
        seed_map = {}
        for asset_name, group in frame.groupby("Asset", sort=False):
            series = group.copy()
            series["Refit Date"] = pd.to_datetime(series["Refit Date"])
            seed_map[asset_name] = series.set_index("Refit Date")["Best Lambda"].sort_index()
        return seed_map

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "lambdas" in payload:
        return {
            asset_name: pd.Series(asset_lambdas).sort_index()
            for asset_name, asset_lambdas in payload["lambdas"].items()
        }

    if isinstance(payload, dict):
        return {
            asset_name: pd.Series(asset_lambdas).sort_index()
            for asset_name, asset_lambdas in payload.items()
        }

    raise RuntimeError("Unsupported lambda seed format.")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    lambda_seed = Path(args.lambda_seed).expanduser().resolve() if args.lambda_seed else None

    print("=" * 72)
    print("SHU, YU, MULVEY (2025) REPRODUCTION")
    print("Dynamic Asset Allocation with Asset-Specific Regime Forecasts")
    print("=" * 72)

    prices, returns, ex_returns, macro, rf_daily = load_all_data(
        ASSETS,
        PROXY_FUNDS,
        start=args.start,
        end=args.end,
        cache_dir=cache_dir,
    )

    run = run_model(
        prices,
        returns,
        ex_returns,
        macro,
        rf_daily,
        cache_dir=cache_dir,
        seed_lambda_map=load_lambda_seed(lambda_seed) if lambda_seed else None,
        max_workers=args.workers,
    )
    save_artifacts(run, artifacts_dir)

    print("\nPerformance summary")
    print(format_stats_for_console(run.stats).to_string())

    payload = {
        "data_start": args.start,
        "data_end": args.end,
        "cache_dir": str(cache_dir),
        "artifacts_dir": str(artifacts_dir),
        "workers": args.workers,
        "strategies": run.stats.reset_index().to_dict(orient="records"),
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nArtifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
