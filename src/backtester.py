"""Backtest de carteiras buy-and-hold entre rebalanceamentos periódicos."""

from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

def _validated_target_weights(row: pd.Series, columns: pd.Index) -> np.ndarray:
    target = row.reindex(columns).fillna(0.0).to_numpy(dtype=float)
    if not np.all(np.isfinite(target)) or (target < -1e-8).any():
        raise ValueError("Pesos inválidos no rebalanceamento.")
    target = np.maximum(target, 0.0)
    if target.sum() <= 0:
        raise ValueError("A carteira não possui pesos positivos.")
    if not np.isclose(target.sum(), 1.0, atol=1e-6):
        raise ValueError("Os pesos do rebalanceamento devem somar 1.")
    return target / target.sum()


def backtest_strategy(
    daily_returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """Simula retornos simples, drift dos pesos e custos cobrados uma única vez."""
    if weights.empty:
        raise ValueError("A tabela de pesos está vazia.")
    if not 0 <= transaction_cost < 1:
        raise ValueError("transaction_cost deve estar no intervalo [0, 1).")

    returns = daily_returns.sort_index().loc[daily_returns.index >= weights.index[0]].copy()
    if returns.empty:
        raise ValueError("Não há retornos após o primeiro rebalanceamento.")

    target_by_date = weights.sort_index()
    first_target = _validated_target_weights(target_by_date.iloc[0], returns.columns)
    holdings = initial_capital * first_target
    portfolio_value = float(initial_capital)
    records = []

    for date, row in returns.iterrows():
        value_before = portfolio_value
        transaction_cost_value = 0.0
        turnover = 0.0
        asset_returns = row.to_numpy(dtype=float)
        missing_returns = np.isnan(asset_returns)
        held_before_rebalance = holdings > 1e-10

        if date in target_by_date.index and date != target_by_date.index[0]:
            current_weights = holdings / holdings.sum()
            target = _validated_target_weights(target_by_date.loc[date], returns.columns)
            turnover = float(np.abs(target - current_weights).sum())
            transaction_cost_value = portfolio_value * transaction_cost * turnover
            portfolio_value -= transaction_cost_value
            holdings = portfolio_value * target

        held_after_rebalance = holdings > 1e-10
        # Verifica tanto a carteira trazida do dia anterior quanto a carteira-alvo.
        # Assim, uma posição não pode ser liquidada ao último valor conhecido no
        # rebalanceamento sem que o evento seja registrado.
        missing_held = (held_before_rebalance | held_after_rebalance) & missing_returns
        # Uma cotação ausente conserva o valor monetário da posição naquele dia.
        # A ocorrência é registrada e pode reprovar o experimento no validador.
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)
        if (asset_returns <= -1).any():
            raise ValueError(f"Retorno simples menor ou igual a -100% em {date.date()}.")

        holdings *= 1.0 + asset_returns
        portfolio_value = float(holdings.sum())
        net_return = portfolio_value / value_before - 1.0
        records.append(
            {
                "portfolio_value": portfolio_value,
                "daily_return": net_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost_value,
                "stale_positions": int(missing_held.sum()),
            }
        )

    return pd.DataFrame(records, index=returns.index)


def compute_drawdown_series(
    portfolio_value: pd.Series, initial_capital: Optional[float] = None
) -> pd.Series:
    rolling_max = portfolio_value.cummax()
    if initial_capital is not None:
        rolling_max = rolling_max.clip(lower=float(initial_capital))
    return (portfolio_value - rolling_max) / rolling_max


def maximum_drawdown(
    portfolio_value: pd.Series, initial_capital: Optional[float] = None
) -> float:
    return float(compute_drawdown_series(portfolio_value, initial_capital).min())


def calculate_metrics(
    portfolio_value: pd.Series,
    daily_returns: pd.Series,
    initial_capital: Optional[float] = None,
) -> Dict:
    if portfolio_value.empty or daily_returns.empty:
        raise ValueError("Séries vazias não permitem calcular métricas.")
    if initial_capital is None:
        initial_capital = float(portfolio_value.iloc[0] / (1 + daily_returns.iloc[0]))

    total_return = float(portfolio_value.iloc[-1] / initial_capital - 1)
    if isinstance(portfolio_value.index, pd.DatetimeIndex):
        elapsed_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days + 1
        years = elapsed_days / 365.0
    else:
        years = len(daily_returns) / 365.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    annualized_volatility = float(daily_returns.std(ddof=1) * np.sqrt(365))
    max_drawdown = maximum_drawdown(portfolio_value, initial_capital)
    return {
        "total_return": total_return,
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
    }


def run_full_backtest(
    daily_returns: pd.DataFrame,
    weights_dict: Dict[str, pd.DataFrame],
    transaction_cost: Optional[float] = None,
    initial_capital: Optional[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    transaction_cost = (
        CONFIG["backtest"]["transaction_cost"]
        if transaction_cost is None
        else transaction_cost
    )
    initial_capital = (
        CONFIG["backtest"]["initial_capital"] if initial_capital is None else initial_capital
    )
    print("\n" + "=" * 60)
    print("FASE 3 — BACKTESTING")
    print("=" * 60)
    results = {}
    metric_rows = []
    for name, strategy_weights in weights_dict.items():
        result = backtest_strategy(
            daily_returns, strategy_weights, transaction_cost, initial_capital
        )
        metrics = calculate_metrics(
            result["portfolio_value"], result["daily_return"], initial_capital
        )
        elapsed_days = (result.index[-1] - result.index[0]).days + 1
        elapsed_years = elapsed_days / 365.0
        metrics["total_turnover"] = float(result["turnover"].sum())
        metrics["annualized_turnover"] = (
            metrics["total_turnover"] / elapsed_years if elapsed_years > 0 else 0.0
        )
        metrics["total_transaction_cost"] = float(result["transaction_cost"].sum())
        metrics["rebalance_events"] = int((result["turnover"] > 1e-12).sum())
        metrics["strategy"] = name
        results[name] = result
        metric_rows.append(metrics)
        print(
            f"{name}: retorno={metrics['total_return']:.2%}, "
            f"vol={metrics['annualized_volatility']:.2%}, "
            f"MDD={metrics['max_drawdown']:.2%}"
        )
    return results, pd.DataFrame(metric_rows).set_index("strategy")


def save_backtest_results(
    results: Dict[str, pd.DataFrame], metrics_df: pd.DataFrame, outdir: Path
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in results.items():
        dataframe.to_csv(outdir / f"backtest_{name}.csv")
    metrics_df.to_csv(outdir / "metrics_summary.csv")
    print(f"[SALVO] Métricas em {outdir / 'metrics_summary.csv'}")
