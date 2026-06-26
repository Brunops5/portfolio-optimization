"""
Módulo de otimização de portfólio com restrição de correlação média máxima.

Framework: PyPortfolioOpt (pypfopt) para cálculo de inputs (retornos esperados,
matriz de covariância, bounds e estrutura de restrições). A otimização propriamente
dita é realizada via scipy.optimize (SLSQP) para incorporar a restrição customizada
de correlação média máxima — uma extensão ao modelo mean-variance padrão.

Referências:
    - Martin, R. A. (2021). PyPortfolioOpt: portfolio optimisation in Python.
    - Markowitz, H. (1952). Portfolio Selection.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple

# PyPortfolioOpt — framework base para otimização de portfólio
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RISK_FREE_DAILY = CONFIG["optimization"]["risk_free_rate_annual"] / 365


def weighted_average_correlation(weights: np.ndarray, corr_matrix: np.ndarray) -> float:
    """
    Calcula a correlação média ponderada de uma carteira.

    Fórmula:
        ρ̄(w) = Σ_i Σ_{j>i} w_i w_j ρ_{ij} / Σ_i Σ_{j>i} w_i w_j

    Args:
        weights: Vetor de pesos (n_assets,).
        corr_matrix: Matriz de correlação de Pearson (n_assets, n_assets).

    Returns:
        Correlação média ponderada (float entre -1 e 1).
    """
    n = len(weights)
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            prod = weights[i] * weights[j]
            numerator += prod * corr_matrix[i, j]
            denominator += prod
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def negative_sharpe_pypfopt(
    weights: np.ndarray,
    mu: np.ndarray,
    S: np.ndarray,
    risk_free: float = RISK_FREE_DAILY,
) -> float:
    """
    Função objetivo: negativo do Sharpe Ratio anualizado (para minimização).

    Usa as convenções do PyPortfolioOpt:
        - mu: retornos esperados anualizados
        - S: matriz de covariância anualizada
        - weights: pesos do portfólio
        - risk_free: taxa livre de risco anualizada

    Como mu e S já são anualizados pelo pypfopt, o Sharpe calculado aqui
    já é anualizado. O termo risk_free * 365 equivale à taxa livre de risco
    anualizada quando risk_free é a taxa diária.
    """
    port_return = np.dot(weights, mu)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    if port_volatility < 1e-12:
        return 0.0
    return -(port_return - risk_free * 365) / port_volatility


def portfolio_variance_pypfopt(weights: np.ndarray, S: np.ndarray) -> float:
    """Função objetivo: variância anualizada do portfólio (convenção pypfopt)."""
    return np.dot(weights.T, np.dot(S, weights))


def optimize_portfolio(
    returns: pd.DataFrame,
    max_avg_correlation: float,
    objective: str = "max_sharpe",
    min_weight: float = 0.0,
    max_weight: float = 0.40,
    risk_free_annual: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Otimiza um portfólio com restrição de correlação média máxima.

    Utiliza PyPortfolioOpt para cálculo dos inputs (retornos esperados e
    covariância anualizados) e scipy.optimize (SLSQP) para a otimização com
    a restrição customizada de correlação média — uma extensão ao modelo
    mean-variance clássico.

    Args:
        returns: DataFrame de retornos diários da janela de estimação.
        max_avg_correlation: Limite superior para a correlação média.
        objective: "max_sharpe" ou "min_volatility".
        min_weight: Peso mínimo por ativo.
        max_weight: Peso máximo por ativo.
        risk_free_annual: Taxa livre de risco anualizada.

    Returns:
        weights: Vetor de pesos ótimos.
        info: Dict com métricas da otimização.
    """
    n_assets = returns.shape[1]
    assets = returns.columns.tolist()

    if risk_free_annual is None:
        risk_free_annual = CONFIG["optimization"]["risk_free_rate_annual"]
    risk_free_daily = risk_free_annual / 365


    # PyPortfolioOpt: cálculo dos inputs da otimização
    
    # Recupera preços a partir de retornos logarítmicos para usar as
    # funções do pypfopt (que esperam séries de preços).
    # Retornos log: r_t = ln(P_t / P_{t-1})  =>  P_t = P_0 * exp(cumsum(r_t))
    # Usamos P_0 = 1.0 para todos os ativos (arbitrário para covariância).
    prices = np.exp(returns.cumsum())
    prices.iloc[0] = 1.0  # Garante preço inicial unitário

    # Retornos esperados anualizados (mean historical return)
    mu = expected_returns.mean_historical_return(prices, frequency=365).values

    # Matriz de covariância anualizada (amostral)
    S = risk_models.sample_cov(prices, frequency=365).values

    # Matriz de correlação de Pearson
    corr_matrix = returns.corr().values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(corr_matrix, 1.0)

    
    # Restrições scipy.optimize
   
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # soma = 1
    ]

    def corr_constraint(w):
        return max_avg_correlation - weighted_average_correlation(w, corr_matrix)

    constraints.append({"type": "ineq", "fun": corr_constraint})

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    x0 = np.array([1.0 / n_assets] * n_assets)

    if objective == "max_sharpe":
        obj_fun = lambda w: negative_sharpe_pypfopt(w, mu, S, risk_free_daily)
    elif objective == "min_volatility":
        obj_fun = lambda w: portfolio_variance_pypfopt(w, S)
    else:
        raise ValueError(f"Objetivo '{objective}' não suportado.")

    result = minimize(
        obj_fun,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        # Fallback: otimização sem restrição de correlação (relaxada)
        print(
            f"[AVISO] Otimização com corr ≤ {max_avg_correlation} falhou. "
            f"Tentando sem restrição de correlação..."
        )
        result = minimize(
            obj_fun,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraints[0]],  # apenas soma = 1
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )

    weights = result.x
    weights = np.maximum(weights, 0)
    weights /= weights.sum()

    # Métricas da carteira resultante (usando mu e S anualizados do pypfopt)
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    avg_corr = weighted_average_correlation(weights, corr_matrix)

    info = {
        "success": result.success,
        "message": result.message,
        "expected_daily_return": float(port_return / 365),
        "expected_annual_return": float(port_return),
        "daily_volatility": float(port_vol / np.sqrt(365)),
        "annual_volatility": float(port_vol),
        "sharpe_ratio": float(
            (port_return - risk_free_annual) / port_vol if port_vol > 1e-12 else 0.0
        ),
        "avg_correlation": float(avg_corr),
        "num_assets_selected": int((weights > 1e-6).sum()),
    }

    return weights, info


def run_rolling_optimization(
    daily_returns: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    profiles: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Executa otimização mensal com janela deslizante de 12 meses.

    Args:
        daily_returns: Retornos diários completos.
        monthly_returns: Retornos mensais completos (para alinhamento de datas).
        profiles: Dict de perfis do config.yaml.

    Returns:
        Dict com DataFrames de pesos para cada perfil + benchmark.
    """
    if profiles is None:
        profiles = CONFIG["profiles"]

    window_months = CONFIG["estimation_window_months"]
    months = monthly_returns.index
    all_assets = daily_returns.columns.tolist()

    all_weights = {name: [] for name in list(profiles.keys()) + ["benchmark"]}
    rebalance_dates = []

    for i in range(window_months, len(months)):
        rebalance_date = months[i]
        window_start = months[i - window_months]
        window_end = months[i - 1]

        # Seleciona dados diários dentro da janela
        mask = (daily_returns.index >= window_start) & (daily_returns.index <= window_end)
        window_returns = daily_returns.loc[mask].dropna(how="all", axis=1)
        # Remove ativos com menos de 60 observações na janela
        window_returns = window_returns.loc[:, window_returns.count() >= 60]

        if window_returns.empty or window_returns.shape[1] < 2:
            print(f"[PULAR] Dados insuficientes em {rebalance_date.date()}")
            continue

        rebalance_dates.append(rebalance_date)
        assets_in_window = window_returns.columns.tolist()
        print(
            f"\n[REBALANCEAMENTO] {rebalance_date.date()} | "
            f"Janela: {window_start.date()} a {window_end.date()} | "
            f"Ativos: {assets_in_window}"
        )

        for profile_key, profile_cfg in profiles.items():
            try:
                w, info = optimize_portfolio(
                    window_returns,
                    max_avg_correlation=profile_cfg["max_avg_correlation"],
                    objective=CONFIG["optimization"]["objective"],
                    min_weight=CONFIG["optimization"]["min_weight"],
                    max_weight=CONFIG["optimization"]["max_weight"],
                )
                print(
                    f"  {profile_cfg['name']:12s}: Sharpe={info['sharpe_ratio']:.3f}, "
                    f"AvgCorr={info['avg_correlation']:.3f}, "
                    f"Ativos={info['num_assets_selected']}"
                )
                # Alinha com all_assets (preenche zero para ativos fora da janela)
                w_aligned = pd.Series(0.0, index=all_assets)
                w_aligned[assets_in_window] = w
                all_weights[profile_key].append(w_aligned.values)
            except Exception as e:
                print(f"  [ERRO] {profile_cfg['name']}: {e}")
                all_weights[profile_key].append(np.full(len(all_assets), np.nan))

        # Benchmark: alocação igualitária entre ativos disponíveis na janela
        bench_w = pd.Series(0.0, index=all_assets)
        bench_w[assets_in_window] = 1.0 / len(assets_in_window)
        all_weights["benchmark"].append(bench_w.values)

    # Constrói DataFrames de pesos com colunas fixas (all_assets)
    result = {}
    for key in list(profiles.keys()) + ["benchmark"]:
        if not all_weights[key]:
            continue
        df = pd.DataFrame(
            all_weights[key],
            index=rebalance_dates,
            columns=all_assets,
        )
        result[key] = df

    return result


def save_weights(weights_dict: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """Persiste os pesos em CSV."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in weights_dict.items():
        path = outdir / f"weights_{name}.csv"
        df.to_csv(path)
        print(f"[SALVO] Pesos {name}: {path}")


if __name__ == "__main__":
    # Teste rápido com dados dummy
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="B")
    assets = ["BTC", "ETH", "SOL"]
    rets = pd.DataFrame(
        np.random.normal(0.001, 0.03, (len(dates), 3)),
        index=dates,
        columns=assets,
    )
    w, info = optimize_portfolio(rets, max_avg_correlation=0.70)
    print("Pesos:", dict(zip(assets, np.round(w, 4))))
    print("Info:", info)
