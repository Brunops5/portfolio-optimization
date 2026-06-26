"""
Módulo de backtesting com janelas deslizantes mensais.
Calcula evolução do patrimônio, métricas de desempenho e drawdowns.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RESULTS_DIR = Path(CONFIG["paths"]["results"]) / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def backtest_strategy(
    daily_returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """
    Simula a evolução de uma estratégia com rebalanceamento mensal.

    Args:
        daily_returns: Retornos logarítmicos diários de todos os ativos.
        weights: DataFrame de pesos por data de rebalanceamento.
        transaction_cost: Custo proporcional por rebalanceamento.
        initial_capital: Patrimônio inicial.

    Returns:
        DataFrame com colunas: ['portfolio_value', 'daily_return']
        index: datas diárias do período de backtest.
    """
    rebalance_dates = weights.index
    start_date = rebalance_dates[0]
    end_date = daily_returns.index[-1]

    # Filtra retornos do período de backtest
    mask = (daily_returns.index >= start_date) & (daily_returns.index <= end_date)
    rets = daily_returns.loc[mask].copy()

    portfolio_value = initial_capital
    values = []
    daily_rets = []
    current_weights = weights.iloc[0].values

    for date, row in rets.iterrows():
        # Verifica se é data de rebalanceamento
        if date in weights.index:
            new_weights = weights.loc[date].reindex(rets.columns).fillna(0).values
            new_weights = np.maximum(new_weights, 0)
            s = new_weights.sum()
            if s > 0:
                new_weights /= s

            # Custo de transação proporcional ao turnover
            turnover = np.sum(np.abs(new_weights - current_weights))
            cost = turnover * transaction_cost
            portfolio_value *= (1 - cost)
            current_weights = new_weights

        # Retorno do portfólio no dia
        day_ret = np.dot(current_weights, row.fillna(0).values)
        portfolio_value *= np.exp(day_ret)

        values.append(portfolio_value)
        daily_rets.append(day_ret)

    result = pd.DataFrame(
        {"portfolio_value": values, "daily_return": daily_rets},
        index=rets.index,
    )
    return result


def compute_drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    """Calcula a série de drawdown percentual."""
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    return drawdown


def maximum_drawdown(portfolio_value: pd.Series) -> float:
    """Retorna o máximo drawdown (valor negativo)."""
    return compute_drawdown_series(portfolio_value).min()


def calculate_metrics(portfolio_value: pd.Series, daily_returns: pd.Series) -> Dict:
    """
    Calcula métricas de desempenho padrão.

    Returns:
        Dict com: total_return, annualized_return, annualized_volatility,
                  sharpe_ratio, max_drawdown, calmar_ratio.
    """
    rf_annual = CONFIG["optimization"]["risk_free_rate_annual"]
    n_years = len(daily_returns) / 365.0

    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    # Retorno anualizado a partir de log-retornos diários (média geométrica).
    # Consistente com a convenção de log-retornos adotada no projeto.
    # Para criptoativos, o mercado opera todos os dias, portanto usa-se 365 dias.
    annualized_return = np.exp(daily_returns.mean() * 365.0) - 1 if n_years > 0 else 0.0
    annualized_volatility = daily_returns.std() * np.sqrt(365)

    if annualized_volatility > 1e-12:
        sharpe = (annualized_return - rf_annual) / annualized_volatility
    else:
        sharpe = 0.0

    mdd = maximum_drawdown(portfolio_value)

    if abs(mdd) > 1e-12:
        calmar = annualized_return / abs(mdd)
    else:
        calmar = np.inf

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(mdd),
        "calmar_ratio": float(calmar),
    }


def run_full_backtest(
    daily_returns: pd.DataFrame,
    weights_dict: Dict[str, pd.DataFrame],
    transaction_cost: Optional[float] = None,
    initial_capital: Optional[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Executa backtest para todas as estratégias.

    Returns:
        (results_dict, metrics_df)
    """
    if transaction_cost is None:
        transaction_cost = CONFIG["backtest"]["transaction_cost"]
    if initial_capital is None:
        initial_capital = CONFIG["backtest"]["initial_capital"]

    print("\n" + "=" * 60)
    print("FASE 3 — BACKTESTING")
    print("=" * 60)

    results = {}
    metrics_rows = []

    for name, weights in weights_dict.items():
        print(f"\n[BACKTEST] Estratégia: {name}")
        try:
            bt = backtest_strategy(daily_returns, weights, transaction_cost, initial_capital)
            metrics = calculate_metrics(bt["portfolio_value"], bt["daily_return"])
            metrics["strategy"] = name
            metrics_rows.append(metrics)
            results[name] = bt

            print(f"  Retorno Total:      {metrics['total_return']*100:.2f}%")
            print(f"  Retorno Anualizado: {metrics['annualized_return']*100:.2f}%")
            print(f"  Volatilidade Anual: {metrics['annualized_volatility']*100:.2f}%")
            print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
        except Exception as e:
            print(f"  [ERRO] {e}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.set_index("strategy")
    return results, metrics_df


def save_backtest_results(results: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame, outdir: Path) -> None:
    """Persiste resultados do backtesting."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        path = outdir / f"backtest_{name}.csv"
        df.to_csv(path)

    metrics_path = outdir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\n[SALVO] Métricas em {metrics_path}")


if __name__ == "__main__":
    # Teste dummy
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    rets = pd.DataFrame(np.random.normal(0.0005, 0.02, (len(dates), 3)), index=dates, columns=["A", "B", "C"])
    w_dates = pd.date_range("2023-02-01", "2023-12-01", freq="MS")
    weights = pd.DataFrame(np.full((len(w_dates), 3), 1/3), index=w_dates, columns=["A", "B", "C"])
    bt = backtest_strategy(rets, weights)
    print(bt.tail())
    print(calculate_metrics(bt["portfolio_value"], bt["daily_return"]))
