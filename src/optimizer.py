"""Otimização rolling de Markowitz orientada a perfis de volatilidade."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pypfopt import EfficientFrontier, expected_returns, risk_models

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
ANNUALIZATION_FACTOR = 365


def load_constituent_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Carrega e valida a composição point-in-time do NCI."""
    path = path or CONFIG_PATH.parent / CONFIG["paths"]["constituent_history"]
    with Path(path).open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)

    rows = document.get("schedule", []) if isinstance(document, dict) else []
    if not rows:
        raise ValueError("O histórico de constituintes do NCI está vazio.")
    history = pd.DataFrame(rows)
    required_columns = {"effective_date", "quarter", "constituents"}
    if not required_columns.issubset(history.columns):
        raise ValueError("Histórico do NCI sem os campos obrigatórios.")
    history["effective_date"] = pd.to_datetime(history["effective_date"])
    if history["effective_date"].duplicated().any():
        raise ValueError("Datas efetivas duplicadas no histórico do NCI.")
    history = history.sort_values("effective_date").set_index("effective_date")
    history["constituents"] = history["constituents"].map(tuple)

    historical_assets = {
        asset for constituents in history["constituents"] for asset in constituents
    }
    missing_assets = historical_assets - set(CONFIG["assets"])
    if missing_assets:
        raise ValueError(
            "Constituintes históricos ausentes de config.assets: "
            f"{sorted(missing_assets)}"
        )
    return history


def constituent_record_for_date(history: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Retorna o último universo do NCI que já era efetivo na data."""
    available = history.loc[history.index <= pd.Timestamp(date)]
    if available.empty:
        raise ValueError(f"Não há composição do NCI vigente em {date}.")
    return available.iloc[-1]


def estimate_inputs(simple_returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """Estima retorno e covariância anualizados com as funções padrão da lib.

    Usa ``expected_returns.mean_historical_return`` e ``risk_models.sample_cov``
    do PyPortfolioOpt com os argumentos padrão; apenas ``frequency=365`` é fixado
    explicitamente, seguindo a convenção de anualização do projeto para mercados
    que operam continuamente. Em seguida, a covariância passa por
    ``risk_models.fix_nonpositive_semidefinite`` (método espectral), garantia
    numérica exigida metodologicamente; os autovalores extremos antes e depois
    da correção ficam registrados no diagnóstico.
    """
    if simple_returns.empty or simple_returns.shape[1] < 2:
        raise ValueError("São necessários retornos de pelo menos dois ativos.")
    if simple_returns.isna().any().any():
        raise ValueError("A janela de estimação contém valores ausentes.")

    mu = expected_returns.mean_historical_return(
        simple_returns,
        returns_data=True,
        frequency=ANNUALIZATION_FACTOR,
    )
    # ``sample_cov`` já corrige matrizes não PSD internamente. A matriz bruta é
    # calculada separadamente apenas para tornar o diagnóstico antes/depois real.
    raw_covariance = simple_returns.cov() * ANNUALIZATION_FACTOR
    min_eigenvalue_before = float(np.linalg.eigvalsh(raw_covariance.values).min())
    covariance = risk_models.sample_cov(
        simple_returns,
        returns_data=True,
        frequency=ANNUALIZATION_FACTOR,
        fix_method="spectral",
    )
    min_eigenvalue_after = float(np.linalg.eigvalsh(covariance.values).min())

    diagnostics = {
        "min_eigenvalue_before": min_eigenvalue_before,
        "min_eigenvalue_after": min_eigenvalue_after,
        "covariance_psd": min_eigenvalue_after >= -1e-10,
    }
    return mu, covariance, diagnostics


def optimize_portfolio(
    simple_returns: pd.DataFrame,
    target_volatility: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Maximiza retorno esperado sob um limite anual de volatilidade.

    Quando o alvo é inviável, retorna a carteira de mínima volatilidade e sinaliza
    explicitamente ``target_feasible=False``. Isso mantém o backtest executável sem
    esconder que o perfil não pôde ser atendido naquela janela.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility deve ser positivo.")
    if not 0 <= min_weight <= max_weight <= 1:
        raise ValueError("Os limites de peso devem satisfazer 0 <= min <= max <= 1.")

    mu, covariance, covariance_info = estimate_inputs(simple_returns)
    bounds = (min_weight, max_weight)

    minimum_frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)
    minimum_frontier.min_volatility()
    minimum_volatility = float(minimum_frontier.portfolio_performance()[1])

    target_feasible = target_volatility + 1e-7 >= minimum_volatility
    if target_feasible:
        frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)
        frontier.efficient_risk(target_volatility=float(target_volatility))
        weights = frontier.weights
        message = "ok"
    else:
        weights = minimum_frontier.weights
        message = (
            f"alvo inviável ({target_volatility:.4f}); "
            f"mínima volatilidade={minimum_volatility:.4f}"
        )

    weights = np.asarray(weights, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()

    portfolio_return = float(weights @ mu.values)
    portfolio_volatility = float(np.sqrt(weights @ covariance.values @ weights))
    info = {
        "success": True,
        "message": message,
        "objective": "max_return_for_target_volatility",
        "target_volatility": float(target_volatility),
        "target_feasible": bool(target_feasible),
        "minimum_volatility": minimum_volatility,
        "expected_annual_return": portfolio_return,
        "annual_volatility": portfolio_volatility,
        "num_assets_selected": int((weights > 1e-6).sum()),
        "max_weight_observed": float(weights.max()),
        **covariance_info,
    }
    return weights, info


def _rebalance_dates(
    daily_returns: pd.DataFrame, window_months: int, frequency: str
) -> pd.DatetimeIndex:
    """Primeiro dia observado de cada período após a janela inicial."""
    observed_dates = pd.Series(daily_returns.index, index=daily_returns.index)
    # ``closed='left'`` faz W-MON representar [segunda, segunda seguinte),
    # evitando o comportamento padrão do pandas, que agruparia terça a segunda.
    first_observed_by_period = (
        observed_dates.resample(frequency, closed="left", label="left").min().dropna()
    )
    first_allowed = (
        daily_returns.index.min().to_period("M") + window_months
    ).start_time
    return pd.DatetimeIndex(
        first_observed_by_period.loc[first_observed_by_period >= first_allowed].to_numpy()
    )


def prepare_estimation_window(
    daily_returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    constituent_history: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Constrói a janela e o universo elegível usados por cálculo e relatório."""
    constituent_record = constituent_record_for_date(constituent_history, rebalance_date)
    active_constituents = list(constituent_record["constituents"])
    missing_columns = set(active_constituents) - set(daily_returns.columns)
    if missing_columns:
        raise ValueError(
            f"Constituintes do NCI sem série em {rebalance_date.date()}: "
            f"{sorted(missing_columns)}"
        )

    window_start = rebalance_date - pd.DateOffset(
        months=CONFIG["estimation_window_months"]
    )
    window = daily_returns.loc[
        (daily_returns.index >= window_start) & (daily_returns.index < rebalance_date)
    ]
    recent_cutoff = rebalance_date - pd.Timedelta(
        days=CONFIG["optimization"]["max_data_gap_days"]
    )
    eligible = []
    for asset in active_constituents:
        observations = window[asset].dropna()
        if (
            len(observations) >= CONFIG["optimization"]["min_observations"]
            and observations.index.max() >= recent_cutoff
        ):
            eligible.append(asset)
    return window[eligible].dropna(how="any"), constituent_record, eligible


def run_rolling_optimization(
    daily_returns: pd.DataFrame,
    profiles: Optional[Dict] = None,
    constituent_history: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Otimiza no início de cada período usando somente a janela anterior."""
    if profiles is None:
        profiles = CONFIG["profiles"]
    if constituent_history is None:
        constituent_history = load_constituent_history()

    opt_cfg = CONFIG["optimization"]
    window_months = CONFIG["estimation_window_months"]
    rebalance_frequency = CONFIG["rebalance_frequency"]
    min_observations = opt_cfg["min_observations"]
    all_assets = daily_returns.columns.tolist()

    by_strategy = {name: [] for name in [*profiles, "benchmark"]}
    accepted_dates = []
    diagnostics = []

    for rebalance_date in _rebalance_dates(
        daily_returns, window_months, rebalance_frequency
    ):
        estimation_returns, constituent_record, eligible = prepare_estimation_window(
            daily_returns, rebalance_date, constituent_history
        )
        active_constituents = list(constituent_record["constituents"])
        window_start = rebalance_date - pd.DateOffset(months=window_months)
        if estimation_returns.shape[1] < 2 or len(estimation_returns) < min_observations:
            print(f"[PULAR] Dados insuficientes em {rebalance_date.date()}")
            continue

        accepted_dates.append(rebalance_date)
        print(
            f"\n[REBALANCEAMENTO] {rebalance_date.date()} | "
            f"Janela: {window_start.date()} a {(rebalance_date - pd.Timedelta(days=1)).date()} | "
            f"Ativos: {eligible}"
        )
        for profile_key, profile_cfg in profiles.items():
            target_volatility = profile_cfg["target_volatility"]
            profile_name = profile_cfg["name"]
            try:
                weights, info = optimize_portfolio(
                    estimation_returns,
                    target_volatility=target_volatility,
                    min_weight=opt_cfg["min_weight"],
                    max_weight=opt_cfg["max_weight"],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao otimizar {profile_name} em {rebalance_date.date()}: {exc}"
                ) from exc

            aligned = pd.Series(0.0, index=all_assets)
            aligned[eligible] = weights
            by_strategy[profile_key].append(aligned.values)
            diagnostics.append(
                {
                    "rebalance_date": rebalance_date,
                    "strategy": profile_key,
                    "observations": len(estimation_returns),
                    "nci_effective_date": constituent_record.name,
                    "nci_quarter": constituent_record["quarter"],
                    "nci_constituent_count": len(active_constituents),
                    "eligible_count": len(eligible),
                    "excluded_constituents": "|".join(
                        sorted(set(active_constituents) - set(eligible))
                    ),
                    **info,
                }
            )
            status = "" if info["target_feasible"] else " [ALVO INVIÁVEL]"
            print(
                f"  {profile_name:12s}: "
                f"vol={info['annual_volatility']:.3f}/{info['target_volatility']:.3f}, "
                f"ativos={info['num_assets_selected']}{status}"
            )

        benchmark = pd.Series(0.0, index=all_assets)
        benchmark[eligible] = 1.0 / len(eligible)
        by_strategy["benchmark"].append(benchmark.values)

    result = {
        name: pd.DataFrame(values, index=accepted_dates, columns=all_assets)
        for name, values in by_strategy.items()
        if values
    }
    return result, pd.DataFrame(diagnostics)


def save_weights(weights_dict: Dict[str, pd.DataFrame], outdir: Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in weights_dict.items():
        path = outdir / f"weights_{name}.csv"
        dataframe.to_csv(path)
        print(f"[SALVO] Pesos {name}: {path}")


def save_optimization_diagnostics(diagnostics: pd.DataFrame, outdir: Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "optimization_diagnostics.csv"
    diagnostics.to_csv(path, index=False)
    print(f"[SALVO] Diagnósticos da otimização em {path}")
