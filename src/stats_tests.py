"""
Módulo de análise estatística comparativa.
Teste de Wilcoxon de postos sinalizados e estatísticas descritivas.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path
from typing import Dict


def periodic_returns_from_backtest(
    backtest_df: pd.DataFrame, frequency: str
) -> pd.Series:
    """
    Extrai log-retornos periódicos a partir dos retornos simples diários.

    Os retornos são compostos em períodos não sobrepostos e convertidos para
    log-retorno apenas para a comparação estatística.
    """
    if "daily_return" not in backtest_df:
        raise ValueError("O backtest deve conter a coluna daily_return.")
    daily_returns = backtest_df["daily_return"]
    if daily_returns.empty or (daily_returns <= -1).any():
        raise ValueError("Retornos diários inválidos para agregação periódica.")
    resampler = daily_returns.resample(frequency)
    periodic_simple = (1.0 + daily_returns).resample(frequency).prod() - 1.0
    if frequency.upper().startswith("W-"):
        # O mercado cripto opera sete dias. Semanas parciais nos limites da
        # amostra não são comparáveis às demais e ficam fora do teste.
        periodic_simple = periodic_simple.loc[resampler.count() == 7]
    return np.log1p(periodic_simple).dropna()


def wilcoxon_comparison(
    returns_a: pd.Series,
    returns_b: pd.Series,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict:
    """
    Aplica o teste de Wilcoxon pareado entre duas séries de retornos mensais.

    Args:
        returns_a, returns_b: Séries de retornos mensais (mesmo índice temporal).

    Returns:
        Dict com estatísticas do teste e diferenças.
    """
    # Alinha e remove NaN
    df = pd.concat([returns_a, returns_b], axis=1).dropna()
    df.columns = [label_a, label_b]

    if len(df) < 6:
        return {
            "n": len(df),
            "wilcoxon_statistic": np.nan,
            "p_value": np.nan,
            "median_diff": np.nan,
            "mean_diff": np.nan,
            "std_diff": np.nan,
            "min_diff": np.nan,
            "max_diff": np.nan,
            "win_rate_a": np.nan,
            "win_rate_b": np.nan,
        }

    diffs = df[label_a] - df[label_b]

    # Wilcoxon
    if np.allclose(diffs, 0):
        stat, pvalue = 0.0, 1.0
    else:
        try:
            stat, pvalue = wilcoxon(diffs, zero_method="wilcox", correction=False)
        except ValueError:
            stat, pvalue = np.nan, np.nan

    win_a = (diffs > 0).sum()
    win_b = (diffs < 0).sum()
    total = win_a + win_b

    return {
        "n": len(df),
        "wilcoxon_statistic": float(stat),
        "p_value": float(pvalue),
        "median_diff": float(diffs.median()),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std()),
        "min_diff": float(diffs.min()),
        "max_diff": float(diffs.max()),
        "win_rate_a": float(win_a / total) if total > 0 else np.nan,
        "win_rate_b": float(win_b / total) if total > 0 else np.nan,
    }


def compare_all_strategies(
    backtest_results: Dict[str, pd.DataFrame],
    frequency: str = "ME",
) -> pd.DataFrame:
    """
    Executa comparações pareadas Wilcoxon entre todas as estratégias.

    Returns:
        DataFrame com resultados do teste para cada par.
    """
    print("\n" + "=" * 60)
    print(f"FASE 4 — TESTE DE WILCOXON ({frequency})")
    print("=" * 60)

    periodic = {
        name: periodic_returns_from_backtest(df, frequency)
        for name, df in backtest_results.items()
    }
    names = list(periodic.keys())

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stats = wilcoxon_comparison(periodic[a], periodic[b], label_a=a, label_b=b)
            stats["strategy_a"] = a
            stats["strategy_b"] = b
            stats["return_frequency"] = frequency
            rows.append(stats)

            print(f"\n{a} vs {b}:")
            print(
                f"  n={stats['n']}, W={stats['wilcoxon_statistic']:.2f}, "
                f"p={stats['p_value']:.4f}"
            )
            print(
                f"  Diferença média: {stats['mean_diff'] * 100:.2f}%, "
                f"mediana: {stats['median_diff'] * 100:.2f}%"
            )
            print(
                f"  Win rate {a}: {stats['win_rate_a'] * 100:.1f}% | "
                f"{b}: {stats['win_rate_b'] * 100:.1f}%"
            )

    df = pd.DataFrame(rows)
    # Correção de Holm para controlar o erro familiar nas comparações múltiplas.
    valid = df["p_value"].dropna().sort_values()
    adjusted = pd.Series(np.nan, index=df.index, dtype=float)
    running_max = 0.0
    number_of_tests = len(valid)
    for rank, (index, p_value) in enumerate(valid.items()):
        running_max = max(running_max, min(1.0, (number_of_tests - rank) * p_value))
        adjusted.loc[index] = running_max
    df["p_value_holm"] = adjusted
    df = df[[
        "strategy_a", "strategy_b", "return_frequency", "n",
        "wilcoxon_statistic", "p_value", "p_value_holm",
        "mean_diff", "median_diff", "std_diff", "min_diff", "max_diff",
        "win_rate_a", "win_rate_b",
    ]]
    return df


def save_wilcoxon_results(
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "wilcoxon_results",
) -> None:
    """Salva resultados em CSV e Markdown."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SALVO] {csv_path}")

    # Gera tabela Markdown
    md_path = outdir / f"{stem}.md"
    with open(md_path, "w") as f:
        f.write("# Resultados do Teste de Wilcoxon\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"[SALVO] {md_path}")
