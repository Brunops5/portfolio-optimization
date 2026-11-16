#!/usr/bin/env python3
"""Run the PortPerfilCrypto research pipeline."""

import argparse
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import names
from backtester import run_full_backtest, save_backtest_results
from data_loader import main as data_main
from optimizer import (
    run_rolling_optimization,
    save_optimization_diagnostics,
    save_weights,
)
from stats_tests import compare_all_strategies, save_wilcoxon_results
from validators import (
    run_all_validations,
    validate_historical_constituents,
    validate_risk_targets,
    validate_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / names.CONFIG_FILENAME
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def run_pipeline(force_download: bool = False) -> bool:
    """Run each research step and publish outputs only after validation.

    The order matters: prices feed the rolling optimization, optimized weights
    feed the backtest, and validated backtests feed the tables and figures.
    """
    from visualization import generate_all_figures

    results_dir = PROJECT_ROOT / CONFIG[names.PATHS][names.RESULTS]
    reports_dir = PROJECT_ROOT / CONFIG[names.PATHS][names.REPORTS]

    print("=" * 70)
    print("PORTPERFILCRYPTO — CRYPTOCURRENCY PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    data = data_main(force_download=force_download)
    daily_returns = data[names.DAILY_SIMPLE]

    weights_dict, optimization_diagnostics = run_rolling_optimization(daily_returns)

    # Stop before the backtest if any optimization result is invalid.
    prevalidation_errors = []
    prevalidation_errors.extend(validate_weights(weights_dict)[0])
    prevalidation_errors.extend(validate_historical_constituents(weights_dict)[0])
    prevalidation_errors.extend(validate_risk_targets(optimization_diagnostics)[0])
    if prevalidation_errors:
        print("\n[ERROR] Optimization validation failed before the backtest:")
        for error in prevalidation_errors:
            print(f"  - {error}")
        return False

    backtest_results, metrics_df = run_full_backtest(
        daily_returns,
        weights_dict,
    )
    valid = run_all_validations(
        weights_dict,
        backtest_results,
        daily_returns,
        metrics_df,
        CONFIG[names.PROFILES],
        optimization_diagnostics,
    )
    if not valid:
        print("\nFinal artifacts were not published because validation failed.")
        return False

    save_weights(weights_dict, outdir=results_dir / "weights")
    save_optimization_diagnostics(
        optimization_diagnostics,
        outdir=results_dir / "optimization",
    )
    save_backtest_results(
        backtest_results,
        metrics_df,
        outdir=results_dir / "metrics",
    )

    statistics = CONFIG.get(names.STATISTICS, {})
    primary_frequency = statistics.get(
        names.PRIMARY_RETURN_FREQUENCY,
        names.DEFAULT_PRIMARY_FREQUENCY,
    )
    robustness_frequency = statistics.get(names.ROBUSTNESS_RETURN_FREQUENCY)

    primary_results = compare_all_strategies(
        backtest_results,
        frequency=primary_frequency,
    )
    save_wilcoxon_results(
        primary_results,
        outdir=reports_dir / "tables",
        stem="wilcoxon_weekly",
    )

    if robustness_frequency and robustness_frequency != primary_frequency:
        robustness_results = compare_all_strategies(
            backtest_results,
            frequency=robustness_frequency,
        )
        save_wilcoxon_results(
            robustness_results,
            outdir=reports_dir / "tables",
            stem="wilcoxon_monthly_robustness",
        )

    generate_all_figures(
        backtest_results,
        weights_dict,
        daily_returns,
        outdir=reports_dir / "figures",
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY — ALL VALIDATIONS PASSED")
    print("=" * 70)
    print(f"Results: {results_dir}")
    print(f"Reports: {reports_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PortPerfilCrypto portfolio optimization pipeline"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the requested period again instead of using the cache.",
    )
    args = parser.parse_args()

    try:
        success = run_pipeline(force_download=args.force_download)
    except Exception as exc:
        print(f"[CRITICAL ERROR] {exc}", file=sys.stderr)
        return 1

    if success:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
