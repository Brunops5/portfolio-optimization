"""
Módulo de análise estatística comparativa.
Teste de Wilcoxon de postos sinalizados e estatísticas descritivas.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path
from typing import Dict, List, Tuple


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "tables"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def monthly_returns_from_backtest(backtest_df: pd.DataFrame) -> pd.Series:
    """
    Extrai retornos mensais logarítmicos a partir de uma série diária de patrimônio.

    Mantém consistência com o restante do projeto, que opera com log-retornos
    tanto na otimização quanto no backtest diário.

        r_mensal = ln(P_t / P_{t-1})
    """
    pv = backtest_df["portfolio_value"].copy()
    monthly = pv.resample("ME").last()
    monthly_rets = np.log(monthly / monthly.shift(1)).dropna()
    return monthly_rets


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
    try:
        stat, pvalue = wilcoxon(diffs, zero_method="wilcox", correction=False)
    except ValueError:
        # Caso todas as diferenças sejam zero
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
) -> pd.DataFrame:
    """
    Executa comparações pareadas Wilcoxon entre todas as estratégias.

    Returns:
        DataFrame com resultados do teste para cada par.
    """
    print("\n" + "=" * 60)
    print("FASE 4 — TESTE DE WILCOXON")
    print("=" * 60)

    # Extrai retornos mensais
    monthly = {name: monthly_returns_from_backtest(df) for name, df in backtest_results.items()}
    names = list(monthly.keys())

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stats = wilcoxon_comparison(monthly[a], monthly[b], label_a=a, label_b=b)
            stats["strategy_a"] = a
            stats["strategy_b"] = b
            rows.append(stats)

            print(f"\n{a} vs {b}:")
            print(f"  n={stats['n']}, W={stats['wilcoxon_statistic']:.2f}, p={stats['p_value']:.4f}")
            print(f"  Diferença média: {stats['mean_diff']*100:.2f}%, mediana: {stats['median_diff']*100:.2f}%")
            print(f"  Win rate {a}: {stats['win_rate_a']*100:.1f}% | {b}: {stats['win_rate_b']*100:.1f}%")

    df = pd.DataFrame(rows)
    df = df[[
        "strategy_a", "strategy_b", "n", "wilcoxon_statistic", "p_value",
        "mean_diff", "median_diff", "std_diff", "min_diff", "max_diff",
        "win_rate_a", "win_rate_b",
    ]]
    return df


def save_wilcoxon_results(df: pd.DataFrame, outdir: Path = REPORTS_DIR) -> None:
    """Salva resultados em CSV e Markdown."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "wilcoxon_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SALVO] {csv_path}")

    # Gera tabela Markdown
    md_path = outdir / "wilcoxon_results.md"
    with open(md_path, "w") as f:
        f.write("# Resultados do Teste de Wilcoxon\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"[SALVO] {md_path}")


if __name__ == "__main__":
    # Teste dummy
    dates = pd.date_range("2022-01", "2024-12", freq="ME")
    r1 = pd.Series(np.random.normal(0.02, 0.08, len(dates)), index=dates, name="A")
    r2 = pd.Series(np.random.normal(0.01, 0.06, len(dates)), index=dates, name="B")
    print(wilcoxon_comparison(r1, r2))
