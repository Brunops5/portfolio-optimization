"""Validações matemáticas e financeiras do experimento."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from optimizer import constituent_record_for_date, load_constituent_history

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def validate_weights(
    weights_dict: Dict[str, pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    errors = []
    min_allowed = CONFIG["optimization"]["min_weight"]
    max_allowed = CONFIG["optimization"]["max_weight"]
    for name, weights in weights_dict.items():
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            errors.append(f"[{name}] Soma dos pesos diferente de 1.")
        if not np.isfinite(weights.to_numpy()).all():
            errors.append(f"[{name}] Peso não finito detectado.")
        if (weights < min_allowed - 1e-7).any().any():
            errors.append(f"[{name}] Pesos negativos ou abaixo do limite detectados.")
        if (weights > max_allowed + 1e-7).any().any():
            errors.append(f"[{name}] Peso acima do limite {max_allowed:.0%} detectado.")
        if ((weights > 1e-6).sum(axis=1) == 0).any():
            errors.append(f"[{name}] Carteira vazia em alguma data.")
    return errors, []


def validate_historical_constituents(
    weights_dict: Dict[str, pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    """Garante peso zero fora da composição do NCI vigente em cada data."""
    history = load_constituent_history()
    errors = []
    for strategy, weights in weights_dict.items():
        for date, row in weights.iterrows():
            active = set(
                constituent_record_for_date(history, pd.Timestamp(date))["constituents"]
            )
            inactive = row.index.difference(list(active))
            invalid = row.loc[inactive].abs() > 1e-7
            if invalid.any():
                assets = sorted(row.loc[inactive].index[invalid].tolist())
                errors.append(
                    f"[{strategy}] {pd.Timestamp(date).date()}: peso em ativo fora "
                    f"da composição vigente do NCI: {assets}."
                )
    return errors, []


def validate_risk_targets(
    diagnostics: Optional[pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    """Valida PSD, o respeito aos alvos de risco e sinaliza janelas inviáveis."""
    if diagnostics is None or diagnostics.empty:
        return [], ["Diagnóstico de otimização não fornecido."]
    errors = []
    warnings = []
    if not diagnostics["covariance_psd"].fillna(False).all():
        errors.append("Covariância não PSD detectada após a correção.")
    if "success" in diagnostics and not diagnostics["success"].fillna(False).all():
        errors.append("Falha de otimização registrada nos diagnósticos.")
    feasible = diagnostics["target_feasible"].fillna(False)
    violations = diagnostics.loc[
        feasible
        & (
            diagnostics["annual_volatility"]
            > diagnostics["target_volatility"] + 1e-6
        )
    ]
    for _, row in violations.iterrows():
        errors.append(
            f"[{row['strategy']}] {pd.Timestamp(row['rebalance_date']).date()}: "
            "volatilidade excede o alvo declarado."
        )
    infeasible = diagnostics.loc[~diagnostics["target_feasible"].fillna(False)]
    for _, row in infeasible.iterrows():
        warnings.append(
            f"[{row['strategy']}] {pd.Timestamp(row['rebalance_date']).date()}: "
            f"alvo={row['target_volatility']:.3f} inviável; "
            f"mínimo={row['minimum_volatility']:.3f}."
        )
    return errors, warnings


def validate_backtest_consistency(
    backtest_results: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []
    for name, backtest in backtest_results.items():
        if name not in weights_dict:
            errors.append(f"[{name}] Backtest sem tabela de pesos correspondente.")
        if not backtest.index.is_unique or not backtest.index.is_monotonic_increasing:
            errors.append(f"[{name}] Datas do backtest duplicadas ou desordenadas.")
        if not backtest.index.isin(daily_returns.index).all():
            errors.append(f"[{name}] Backtest contém datas ausentes nos retornos.")
        values = backtest["portfolio_value"]
        returns = backtest["daily_return"]
        if (values <= 0).any():
            errors.append(f"[{name}] Patrimônio <= 0 detectado.")
        if returns.max() > 2.0 or returns.min() <= -1.0:
            errors.append(f"[{name}] Retorno diário fora do intervalo plausível.")
        implied = values.pct_change(fill_method=None).dropna()
        recorded = returns.reindex(implied.index).dropna()
        difference = (implied.reindex(recorded.index) - recorded).abs().max()
        if pd.notna(difference) and difference > 1e-10:
            errors.append(
                f"[{name}] Inconsistência entre retorno simples e patrimônio: "
                f"diff={difference:.12f}"
            )
        drawdown = ((values - values.cummax()) / values.cummax()).min()
        if drawdown < -1.0:
            errors.append(f"[{name}] Drawdown abaixo de -100%.")
        if "stale_positions" in backtest and backtest["stale_positions"].sum() > 0:
            message = (
                f"[{name}] {int(backtest['stale_positions'].sum())} posição(ões)-dia "
                "sem nova cotação foram mantidas pelo último valor."
            )
            if CONFIG["backtest"].get("stale_positions_are_errors", True):
                errors.append(message)
            else:
                warnings.append(message)
    missing_backtests = set(weights_dict) - set(backtest_results)
    for name in sorted(missing_backtests):
        errors.append(f"[{name}] Tabela de pesos sem backtest correspondente.")
    return errors, warnings


def validate_metrics(metrics_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    errors = []
    for strategy, row in metrics_df.iterrows():
        if not -1.0 <= row["total_return"] <= 10.0:
            errors.append(f"[{strategy}] Retorno total suspeito.")
        if row["annualized_volatility"] < 1e-6:
            errors.append(f"[{strategy}] Volatilidade anualizada próxima de zero.")
    return errors, []


def run_all_validations(
    weights_dict: Dict[str, pd.DataFrame],
    backtest_results: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    metrics_df: pd.DataFrame,
    profiles: Dict,
    optimization_diagnostics: Optional[pd.DataFrame] = None,
) -> bool:
    expected_strategies = {*profiles, "benchmark"}
    actual_strategies = set(weights_dict)
    strategy_errors = []
    if actual_strategies != expected_strategies:
        strategy_errors.append(
            "Estratégias produzidas diferem das configuradas: "
            f"esperadas={sorted(expected_strategies)}, obtidas={sorted(actual_strategies)}."
        )
    checks = [
        ("Estratégias", strategy_errors, []),
        ("Pesos", *validate_weights(weights_dict)),
        (
            "Constituintes históricos",
            *validate_historical_constituents(weights_dict),
        ),
        ("Alvos de risco e PSD", *validate_risk_targets(optimization_diagnostics)),
        (
            "Backtest",
            *validate_backtest_consistency(backtest_results, weights_dict, daily_returns),
        ),
        ("Métricas", *validate_metrics(metrics_df)),
    ]
    all_errors = []
    warning_count = 0
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO EXPERIMENTO")
    print("=" * 60)
    for name, errors, warnings in checks:
        print(f"\n[{name}]")
        for error in errors[:5]:
            print(f"  ERRO: {error}")
        for warning in warnings[:5]:
            print(f"  AVISO: {warning}")
        if not errors and not warnings:
            print("  Aprovado.")
        all_errors.extend(errors)
        warning_count += len(warnings)
    print(f"\nErros: {len(all_errors)} | Avisos: {warning_count}")
    return not all_errors
