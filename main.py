#!/usr/bin/env python3
"""
Pipeline principal do TCC: Otimização de Portfólios de Criptoativos.
Orquestra: coleta → otimização → backtest → estatísticas → visualização → validação.
"""

import sys
import yaml
from pathlib import Path

# Adiciona src/ ao path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import main as data_main, load_all_assets, compute_returns
from optimizer import run_rolling_optimization, save_weights
from backtester import run_full_backtest, save_backtest_results
from stats_tests import compare_all_strategies, save_wilcoxon_results
from visualization import generate_all_figures
from validators import run_all_validations


def run_pipeline(force_download: bool = False) -> None:
    """Executa o pipeline completo."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("TCC — RECOMENDAÇÃO DE PORTFÓLIOS DE CRIPTOATIVOS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # FASE 1: Dados
    # ------------------------------------------------------------------
    try:
        data = data_main(force_download=force_download)
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha na coleta de dados: {e}")
        sys.exit(1)

    daily_returns = data["daily"]
    monthly_returns = data["monthly"]
    prices = data["prices_filled"]

    # ------------------------------------------------------------------
    # FASE 2: Otimização Rolling
    # ------------------------------------------------------------------
    weights_dict = run_rolling_optimization(daily_returns, monthly_returns)
    save_weights(weights_dict, outdir=Path(config["paths"]["results"]) / "weights")

    # ------------------------------------------------------------------
    # FASE 3: Backtesting
    # ------------------------------------------------------------------
    backtest_results, metrics_df = run_full_backtest(daily_returns, weights_dict)
    save_backtest_results(backtest_results, metrics_df, outdir=Path(config["paths"]["results"]) / "metrics")

    # ------------------------------------------------------------------
    # FASE 4: Estatísticas (Wilcoxon)
    # ------------------------------------------------------------------
    wilcoxon_df = compare_all_strategies(backtest_results)
    save_wilcoxon_results(wilcoxon_df)

    # ------------------------------------------------------------------
    # FASE 5: Visualização
    # ------------------------------------------------------------------
    generate_all_figures(backtest_results, weights_dict, daily_returns)

    # ------------------------------------------------------------------
    # FASE 6: Validação
    # ------------------------------------------------------------------
    valid = run_all_validations(
        weights_dict,
        backtest_results,
        daily_returns,
        metrics_df,
        config["profiles"],
    )

    print("\n" + "=" * 70)
    if valid:
        print("PIPELINE CONCLUÍDO COM SUCESSO — VALIDAÇÃO APROVADA")
    else:
        print("PIPELINE CONCLUÍDO — VALIDAÇÃO APONTOU PROBLEMAS")
    print("=" * 70)
    print(f"Resultados:    {config['paths']['results']}")
    print(f"Relatórios:    {config['paths']['reports']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline TCC — Otimização de Criptoativos")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Força o re-download dos dados da Coinbase.",
    )
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download)
