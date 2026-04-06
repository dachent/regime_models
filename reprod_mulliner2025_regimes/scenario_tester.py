#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_access import DEFAULT_CACHE_DIR, load_all_scenario_sources, load_fama_french_factors
from model_core import run_regime_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario tester for public-data regime model source variants.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Workspace-local cache directory for fetched source data.")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for generated scenario artifacts.")
    parser.add_argument("--quick", action="store_true", help="Run only the core stock-bond correlation and source-consistency checks.")
    return parser.parse_args()


def define_scenarios(sources: dict[str, pd.Series], *, quick: bool = False) -> list[tuple[str, dict[str, pd.Series]]]:
    curve_avg = (sources["gs10_avg"] - sources["tb3ms_avg"]).dropna()
    curve_eom = (sources["tnx_eom"] - sources["irx_eom"]).dropna()

    scenarios: list[tuple[str, dict[str, pd.Series]]] = [
        (
            "Baseline: FRED avg + bond-TR SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources["sb_bondtr"],
            },
        ),
        (
            "H1: AGG splice SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources.get("sb_agg", sources["sb_bondtr"]),
            },
        ),
        (
            "H2: Yield-change SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources["sb_yldchg"],
            },
        ),
        (
            "H3: 5yr rolling window SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources["sb_5yr"],
            },
        ),
        (
            "H5: 5Y Treasury SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources.get("sb_5ybond", sources["sb_bondtr"]),
            },
        ),
        (
            "H6: Inverted sign SB",
            {
                "Market": sources["gspc_m"],
                "Yield_curve": curve_avg,
                "Oil": sources["oil_fred"],
                "Copper": sources["cu_eom"],
                "Monetary_policy": sources["tb3ms_avg"],
                "Volatility": sources["vol_63d"],
                "Stock_bond": sources["sb_inverted"],
            },
        ),
    ]

    if quick:
        return scenarios

    scenarios.extend(
        [
            (
                "EoM yields + bond-TR SB",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_eom,
                    "Oil": sources["oil_fred"],
                    "Copper": sources["cu_eom"],
                    "Monetary_policy": sources["irx_eom"],
                    "Volatility": sources["vol_63d"],
                    "Stock_bond": sources["sb_bondtr"],
                },
            ),
            (
                "EoM yields + yld-chg SB",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_eom,
                    "Oil": sources["oil_fred"],
                    "Copper": sources["cu_eom"],
                    "Monetary_policy": sources["irx_eom"],
                    "Volatility": sources["vol_63d"],
                    "Stock_bond": sources["sb_yldchg"],
                },
            ),
            (
                "21d realized vol",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_avg,
                    "Oil": sources["oil_fred"],
                    "Copper": sources["cu_eom"],
                    "Monetary_policy": sources["tb3ms_avg"],
                    "Volatility": sources["vol_21d"],
                    "Stock_bond": sources["sb_bondtr"],
                },
            ),
        ]
    )

    if "synth_tr_m" in sources:
        scenarios.extend(
            [
                (
                    "SP500TR Market",
                    {
                        "Market": sources["synth_tr_m"],
                        "Yield_curve": curve_avg,
                        "Oil": sources["oil_fred"],
                        "Copper": sources["cu_eom"],
                        "Monetary_policy": sources["tb3ms_avg"],
                        "Volatility": sources["vol_63d"],
                        "Stock_bond": sources["sb_bondtr"],
                    },
                ),
                (
                    "SP500TR + yld-chg SB",
                    {
                        "Market": sources["synth_tr_m"],
                        "Yield_curve": curve_avg,
                        "Oil": sources["oil_fred"],
                        "Copper": sources["cu_eom"],
                        "Monetary_policy": sources["tb3ms_avg"],
                        "Volatility": sources["vol_63d"],
                        "Stock_bond": sources["sb_yldchg"],
                    },
                ),
                (
                    "SP500TR + 63d vol + yld-chg SB",
                    {
                        "Market": sources["synth_tr_m"],
                        "Yield_curve": curve_avg,
                        "Oil": sources["oil_fred"],
                        "Copper": sources["cu_eom"],
                        "Monetary_policy": sources["tb3ms_avg"],
                        "Volatility": sources["vol_63d"],
                        "Stock_bond": sources["sb_yldchg"],
                    },
                ),
            ]
        )

    if "oil_eia_eom" in sources:
        scenarios.extend(
            [
                (
                    "EIA daily oil EoM",
                    {
                        "Market": sources["gspc_m"],
                        "Yield_curve": curve_avg,
                        "Oil": sources["oil_eia_eom"],
                        "Copper": sources["cu_eom"],
                        "Monetary_policy": sources["tb3ms_avg"],
                        "Volatility": sources["vol_63d"],
                        "Stock_bond": sources["sb_bondtr"],
                    },
                ),
                (
                    "Oil splice (FRED+EIA)",
                    {
                        "Market": sources["gspc_m"],
                        "Yield_curve": curve_avg,
                        "Oil": sources["oil_splice"],
                        "Copper": sources["cu_eom"],
                        "Monetary_policy": sources["tb3ms_avg"],
                        "Volatility": sources["vol_63d"],
                        "Stock_bond": sources["sb_bondtr"],
                    },
                ),
            ]
        )

    scenarios.extend(
        [
            (
                "FRED copper only",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_avg,
                    "Oil": sources["oil_fred"],
                    "Copper": sources["cu_fred"],
                    "Monetary_policy": sources["tb3ms_avg"],
                    "Volatility": sources["vol_63d"],
                    "Stock_bond": sources["sb_bondtr"],
                },
            ),
            (
                "Spliced copper (COMEX+FRED)",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_avg,
                    "Oil": sources["oil_fred"],
                    "Copper": sources["cu_splice"],
                    "Monetary_policy": sources["tb3ms_avg"],
                    "Volatility": sources["vol_63d"],
                    "Stock_bond": sources["sb_bondtr"],
                },
            ),
            (
                "All-daily EoM consistent",
                {
                    "Market": sources["gspc_m"],
                    "Yield_curve": curve_eom,
                    "Oil": sources.get("oil_splice", sources["oil_fred"]),
                    "Copper": sources["cu_eom"],
                    "Monetary_policy": sources["irx_eom"],
                    "Volatility": sources["vol_63d"],
                    "Stock_bond": sources["sb_bondtr"],
                },
            ),
        ]
    )

    return scenarios


def evaluate_scenarios(scenarios: list[tuple[str, dict[str, pd.Series]]], factors: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    for idx, (name, econ_variables) in enumerate(scenarios, start=1):
        try:
            _returns, stats, aligned, corr_matrix = run_regime_model(econ_variables, factors)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{idx:2d}/{len(scenarios)}] {name:<55s} FAILED ({exc})")
            continue

        mp_sb = float(corr_matrix.loc["Monetary_policy", "Stock_bond"])
        q1 = float(stats.loc["Q1", "Sharpe"])
        q5 = float(stats.loc["Q5", "Sharpe"])
        lo = float(stats.loc["LO", "Sharpe"])
        diff_sr = float(stats.loc["Q1_Q5", "Sharpe"])
        diff_corr = float(stats.loc["Q1_Q5", "Corr(LO)"])
        diff_t = float(stats.loc["Q1_Q5", "t-stat"])
        n_months = int(stats.loc["Q1_Q5", "N"])
        score = sum(
            [
                abs(0.95 - q1) < 0.10,
                abs(0.17 - q5) < 0.10,
                abs(1.00 - lo) < 0.10,
                abs(0.82 - diff_sr) < 0.20,
                abs(0.37 - diff_corr) < 0.15,
                abs(3.0 - diff_t) < 1.0,
                abs(-0.36 - mp_sb) < 0.15,
            ]
        )
        bar = "#" * score + "." * (7 - score)
        print(
            f"  [{idx:2d}/{len(scenarios)}] {name:<55s} "
            f"MPSB={mp_sb:+.2f} Q1={q1:.2f} Q5={q5:.2f} D={diff_sr:.2f} t={diff_t:.1f} {bar}"
        )
        results.append(
            {
                "name": name,
                "mp_sb": mp_sb,
                "q1": q1,
                "q5": q5,
                "lo": lo,
                "diff_sr": diff_sr,
                "diff_corr": diff_corr,
                "diff_t": diff_t,
                "n": n_months,
                "score": score,
                "aligned_months": len(aligned),
            }
        )

    if not results:
        raise RuntimeError("No scenarios completed successfully.")

    return pd.DataFrame(results).sort_values(["diff_sr", "score"], ascending=[False, False]).reset_index(drop=True)


def save_scenario_chart(results: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = results.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["name"], top["diff_sr"], color="#2a6f97")
    ax.invert_yaxis()
    ax.axvline(0.82, color="#c1121f", linestyle="--", linewidth=1.5, label="Paper Q1-Q5 Sharpe")
    ax.set_title("Top Scenario Q1-Q5 Sharpe")
    ax.set_xlabel("Q1-Q5 Sharpe")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "scenario_q1_q5_sharpe.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    figures_dir = artifacts_dir / "figures"

    print("=" * 80)
    print("REGIME MODEL SCENARIO TESTER")
    print("Mulliner, Harvey, Xia, Fang and Van Hemert (2025)")
    print("=" * 80)

    sources, notes = load_all_scenario_sources(cache_dir=cache_dir)
    factors = load_fama_french_factors(cache_dir=cache_dir)
    scenarios = define_scenarios(sources, quick=args.quick)

    print("\nLoaded source ranges")
    for line in notes:
        print(f"  {line}")
    print(f"\nTotal scenarios to test: {len(scenarios)}")
    if args.quick:
        print("Quick mode enabled: core source-sensitivity scenarios only.")

    results = evaluate_scenarios(scenarios, factors)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "scenario_results.csv", index=False)
    save_scenario_chart(results, figures_dir)

    print("\nTop scenarios")
    print(results[["name", "diff_sr", "diff_t", "mp_sb", "score"]].head(10).to_string(index=False))
    print(f"\nScenario artifacts saved under {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
