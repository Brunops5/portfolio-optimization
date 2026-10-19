#!/usr/bin/env python3
"""Orquestra o pipeline completo do experimento."""

import argparse
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

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
with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def run_pipeline(force_download: bool = False) -> bool:
    """Executa coleta, otimização, backtest, análise e geração de relatórios."""
    from visualization import generate_all_figures

    results_dir = PROJECT_ROOT / CONFIG["paths"]["results"]
    reports_dir = PROJECT_ROOT / CONFIG["paths"]["reports"]

    print("=" * 70)
    print("TCC — RECOMENDAÇÃO DE PORTFÓLIOS DE CRIPTOATIVOS")
    print("=" * 70)

    data = data_main(force_download=force_download)
    daily_returns = data["daily_simple"]

    weights_dict, optimization_diagnostics = run_rolling_optimization(daily_returns)
    prevalidation_errors = []
    prevalidation_errors.extend(validate_weights(weights_dict)[0])
    prevalidation_errors.extend(validate_historical_constituents(weights_dict)[0])
    prevalidation_errors.extend(validate_risk_targets(optimization_diagnostics)[0])
    if prevalidation_errors:
        print("\n[ERRO] Resultados da otimização reprovados antes do backtest:")
        for error in prevalidation_errors:
            print(f"  - {error}")
        return False

    backtest_results, metrics_df = run_full_backtest(daily_returns, weights_dict)
    valid = run_all_validations(
        weights_dict,
        backtest_results,
        daily_returns,
        metrics_df,
        CONFIG["profiles"],
        optimization_diagnostics,
    )
    if not valid:
        print("\nArtefatos finais não foram publicados porque a validação falhou.")
        return False

    save_weights(weights_dict, outdir=results_dir / "weights")
    save_optimization_diagnostics(
        optimization_diagnostics,
        outdir=results_dir / "optimization",
    )
    save_backtest_results(backtest_results, metrics_df, outdir=results_dir / "metrics")

    statistics = CONFIG.get("statistics", {})
    primary_frequency = statistics.get("primary_return_frequency", "W-SUN")
    robustness_frequency = statistics.get("robustness_return_frequency")
    weekly = compare_all_strategies(backtest_results, frequency=primary_frequency)
    save_wilcoxon_results(
        weekly,
        outdir=reports_dir / "tables",
        stem="wilcoxon_weekly",
    )
    if robustness_frequency and robustness_frequency != primary_frequency:
        monthly = compare_all_strategies(
            backtest_results,
            frequency=robustness_frequency,
        )
        save_wilcoxon_results(
            monthly,
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
    print("PIPELINE CONCLUÍDO COM SUCESSO — VALIDAÇÃO APROVADA")
    print("=" * 70)
    print(f"Resultados: {results_dir}")
    print(f"Relatórios: {reports_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline TCC — Otimização de Criptoativos")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignora o cache no intervalo solicitado.",
    )
    args = parser.parse_args()
    try:
        return 0 if run_pipeline(force_download=args.force_download) else 1
    except Exception as exc:
        print(f"[ERRO CRÍTICO] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
