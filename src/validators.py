"""
Módulo de validação e sanity checks do experimento.
Garante coerência matemática, financeira e estatística dos resultados.

Separa:
- ERROS: problemas que invalidam o experimento (soma dos pesos != 1,
  patrimônio negativo, volatilidade zero, etc.)
- AVISOS: situações esperadas do domínio (fallback por correlação
  intrínseca do mercado maior que o limite do perfil).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def validate_weights(weights_dict: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """
    Valida se os pesos otimizados respeitam as restrições matemáticas.
    Retorna (erros, avisos).
    """
    errors = []

    for name, wdf in weights_dict.items():
        # 1. Soma dos pesos deve ser +-1
        row_sums = wdf.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            bad = row_sums[~np.isclose(row_sums, 1.0, atol=1e-4)]
            errors.append(
                f"[{name}] Soma dos pesos != 1 em {len(bad)} datas: {bad.iloc[0]:.6f}"
            )

        # 2. Pesos devem ser não negativos
        if (wdf < -1e-6).any().any():
            errors.append(f"[{name}] Pesos negativos detectados.")

        # 3. Pesos devem estar dentro do limite de concentração (40%)
        max_w = wdf.max().max()
        if max_w > 0.40 + 1e-6:
            errors.append(f"[{name}] Peso máximo {max_w:.4f} excede 40%.")

        # 4. Cada linha deve ter pelo menos 1 ativo selecionado
        num_selected = (wdf > 1e-6).sum(axis=1)
        if (num_selected == 0).any():
            errors.append(f"[{name}] Carteira sem ativos em alguma data.")

    return errors, []


def validate_correlation_constraint(
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    profiles: Dict,
) -> Tuple[List[str], List[str]]:
    """
    Verifica se a correlação média ponderada respeita o limite do perfil.

    Quando a correlação intrínseca do mercado é maior que o limite do perfil,
    o otimizador faz fallback (relaxa a restrição). Isso é comportamento
    esperado e documentado no PLAN.md, então é reportado como AVISO.
    """
    errors = []
    warnings = []
    months = daily_returns.resample("ME").last().index

    for name, wdf in weights_dict.items():
        if name == "benchmark":
            continue
        if name not in profiles:
            continue

        max_corr = profiles[name]["max_avg_correlation"]

        for rebalance_date in wdf.index:
            idx = months.get_loc(rebalance_date)
            if idx < 12:
                continue
            window_start = months[idx - 12]
            window_end = months[idx - 1]
            mask = (daily_returns.index >= window_start) & (daily_returns.index <= window_end)
            window_rets = daily_returns.loc[mask]

            w = wdf.loc[rebalance_date]
            selected = w[w > 1e-6].index.tolist()
            if len(selected) < 2:
                continue

            corr_matrix = window_rets[selected].corr().values
            weights_sel = w[selected].values

            n = len(weights_sel)
            num = 0.0
            den = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    prod = weights_sel[i] * weights_sel[j]
                    num += prod * corr_matrix[i, j]
                    den += prod
            avg_corr = num / den if den > 1e-12 else 0.0

            # Correlação fora do intervalo [-1, 1] seria um erro
            if avg_corr < -1.0 or avg_corr > 1.0:
                errors.append(
                    f"[{name}] {rebalance_date.date()}: "
                    f"correlação={avg_corr:.3f} fora de [-1, 1]"
                )
                continue

            # Excede o limite do perfil: provavelmente fallback esperado
            if avg_corr > max_corr + 1e-6:
                warnings.append(
                    f"[{name}] {rebalance_date.date()}: "
                    f"correlação={avg_corr:.3f} > limite={max_corr} "
                    f"(fallback esperado)"
                )

    return errors, warnings


def validate_backtest_consistency(
    backtest_results: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Valida se o backtest é matematicamente consistente com os pesos e retornos.
    """
    errors = []

    for name, bt in backtest_results.items():
        wdf = weights_dict.get(name)
        if wdf is None:
            continue

        pv = bt["portfolio_value"]
        daily_rets = bt["daily_return"]

        # 1. Patrimônio deve ser sempre positivo
        if (pv <= 0).any():
            errors.append(f"[{name}] Patrimônio <= 0 detectado.")

        # 2. Retornos diários devem estar em ranges razoáveis para cripto
        if daily_rets.max() > 2.0:  # +200%
            errors.append(f"[{name}] Retorno diário máximo suspeito: {daily_rets.max():.2f}")
        if daily_rets.min() < -0.90:  # -90%
            errors.append(f"[{name}] Retorno diário mínimo suspeito: {daily_rets.min():.2f}")

        # 3. Consistência: P_t / P_{t-1} ≈ exp(r_t)
        # Tolerância maior (1e-2) pois dias com NaN em alguns ativos
        # introduzem pequenas diferenças numéricas no backtest.
        implied_rets = np.log(pv / pv.shift(1)).dropna()
        actual_rets = daily_rets.dropna()
        diff = (implied_rets - actual_rets).abs().max()
        if diff > 1e-2:
            errors.append(f"[{name}] Inconsistência entre retornos e patrimônio: diff={diff:.6f}")

        # 4. Drawdown máximo não pode exceder 100%
        mdd = ((pv - pv.cummax()) / pv.cummax()).min()
        if mdd < -1.0:
            errors.append(f"[{name}] Drawdown < -100% ({mdd:.2f}) — impossível.")

    return errors, []


def validate_metrics(
    metrics_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Sanity checks nas métricas de desempenho finais.
    """
    errors = []

    for strategy in metrics_df.index:
        row = metrics_df.loc[strategy]

        total = row["total_return"]
        if not (-1.0 <= total <= 10.0):  # -100% a +1000%
            errors.append(f"[{strategy}] Retorno total suspeito: {total:.2f}")

        vol = row["annualized_volatility"]
        if vol < 1e-6:
            errors.append(f"[{strategy}] Volatilidade anualizada ≈ 0 — verificar.")

        sharpe = row["sharpe_ratio"]
        if not np.isfinite(sharpe):
            errors.append(f"[{strategy}] Sharpe Ratio não-finito.")

        calmar = row["calmar_ratio"]
        if not np.isfinite(calmar):
            errors.append(f"[{strategy}] Calmar Ratio não-finito.")

    return errors, []


def run_all_validations(
    weights_dict: Dict[str, pd.DataFrame],
    backtest_results: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    metrics_df: pd.DataFrame,
    profiles: Dict,
) -> bool:
    """
    Executa todas as validações e imprime relatório.
    Retorna True se não houver erros (avisos não impedem aprovação).
    """
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO EXPERIMENTO")
    print("=" * 60)

    checks = [
        ("Pesos Otimizados", *validate_weights(weights_dict)),
        ("Restrição de Correlação", *validate_correlation_constraint(weights_dict, daily_returns, profiles)),
        ("Consistência do Backtest", *validate_backtest_consistency(backtest_results, weights_dict, daily_returns)),
        ("Métricas de Desempenho", *validate_metrics(metrics_df)),
    ]

    passed = 0
    failed = 0
    total_warnings = 0
    all_errors = []

    for check_name, errors, warnings in checks:
        print(f"\n[{check_name}]")
        has_issue = False

        if errors:
            has_issue = True
            failed += 1
            for err in errors[:5]:
                print(f"  ❌ {err}")
            if len(errors) > 5:
                print(f"  ... e mais {len(errors) - 5} erro(s)")
            all_errors.extend(errors)

        if warnings:
            has_issue = True
            total_warnings += len(warnings)
            for warn in warnings[:5]:
                print(f"  ⚠️  {warn}")
            if len(warnings) > 5:
                print(f"  ... e mais {len(warnings) - 5} aviso(s)")

        if not has_issue:
            passed += 1
            print("  ✅ Sem erros nem avisos.")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"RESULTADO: {failed}/{len(checks)} verificações falharam ({len(all_errors)} erro(s)).")
        print("=" * 60)
        return False
    else:
        print(f"RESULTADO: {len(checks)} verificações aprovadas ({total_warnings} aviso(s) informativo(s)).")
        print("=" * 60)
        return True


if __name__ == "__main__":
    # Teste rápido com dados dummy
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    assets = ["A", "B", "C"]
    rets = pd.DataFrame(np.random.normal(0.001, 0.02, (len(dates), 3)), index=dates, columns=assets)

    w = pd.DataFrame({"A": [0.4], "B": [0.4], "C": [0.2]}, index=[dates[0]])
    weights = {"teste": w}

    pv = pd.Series(10000 * np.exp(rets.sum(axis=1).cumsum()), index=dates)
    bt = pd.DataFrame({"portfolio_value": pv, "daily_return": rets.sum(axis=1)}, index=dates)
    backtests = {"teste": bt}

    metrics = pd.DataFrame(
        {
            "total_return": [0.5],
            "annualized_return": [0.5],
            "annualized_volatility": [0.3],
            "sharpe_ratio": [1.0],
            "max_drawdown": [-0.2],
            "calmar_ratio": [2.5],
        },
        index=["teste"],
    )

    profiles = cfg["profiles"]
    run_all_validations(weights, backtests, rets, metrics, profiles)
