"""
Módulo de visualização e geração de relatórios gráficos.
Estilo orientado a publicação acadêmica — limpo, legível e profissional.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
from pypfopt import EfficientFrontier
from pathlib import Path
from typing import Dict, Optional

from optimizer import estimate_inputs, load_constituent_history, prepare_estimation_window

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

# Paleta institucional — cores sóbrias e distinguíveis
PALETTE = {
    "risco_065": "#1B5E20",     # verde escuro
    "risco_085": "#1565C0",     # azul institucional
    "risco_105": "#C62828",     # vermelho sóbrio
    "benchmark": "#616161",     # cinza médio
}

PUBLICATION_STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": "#BBBBBB",
        "grid.linestyle": "--",
}


def _format_currency_axis(ax, axis="y"):
    """Formata eixo como moeda US$ com separador de milhar."""
    fmt = mticker.StrMethodFormatter("${x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _format_percent_axis(ax, axis="y"):
    """Formata eixo como percentual."""
    fmt = mticker.PercentFormatter()
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------------------------


def plot_portfolio_evolution(
    backtest_results: Dict[str, pd.DataFrame],
    log_scale: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Gráfico de evolução do patrimônio ao longo do tempo.
    Estilo: linhas com área sombreada, escala log opcional.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, df in backtest_results.items():
        color = PALETTE.get(name, "#333333")
        label = name.capitalize()
        x = df.index
        y = df["portfolio_value"]

        # Área sombreada sutil
        ax.fill_between(x, y, alpha=0.08, color=color)
        # Linha principal
        ax.plot(x, y, label=label, linewidth=2.0, color=color, solid_capstyle="round")

        # Anotação do valor final
        if len(y) > 0:
            ax.annotate(
                f"${y.iloc[-1]:,.0f}",
                xy=(x[-1], y.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=11,
                color=color,
                fontweight="bold",
                va="center",
            )

    ax.set_title("Evolução do Patrimônio Líquido por Estratégia", loc="left", pad=15)
    ax.set_xlabel("Data", labelpad=10)
    ax.set_ylabel("Patrimônio (US$)", labelpad=10)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Patrimônio (US$) — Escala Logarítmica", labelpad=10)

    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
    )

    # Formatação de eixos
    _format_currency_axis(ax, axis="y")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    fig.autofmt_xdate(rotation=30)

    # Limites y com margem
    all_vals = pd.concat([df["portfolio_value"] for df in backtest_results.values()])
    ymin, ymax = all_vals.min(), all_vals.max()
    margin = (ymax - ymin) * 0.05
    lower_limit = max(ymin - margin, ymin * 0.8) if log_scale else ymin - margin
    ax.set_ylim(lower_limit, ymax + margin)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[SALVO] {save_path}")
    plt.close()


def plot_drawdowns(
    backtest_results: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Gráfico de drawdown ao longo do tempo.
    Estilo: área preenchida abaixo de zero, linha de referência em 0%.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, df in backtest_results.items():
        color = PALETTE.get(name, "#333333")
        pv = df["portfolio_value"]
        rolling_max = pv.cummax().clip(lower=initial_capital)
        drawdown = (pv - rolling_max) / rolling_max

        ax.fill_between(
            drawdown.index, drawdown * 100, 0,
            alpha=0.18, color=color, linewidth=0
        )
        ax.plot(
            drawdown.index, drawdown * 100,
            label=name.capitalize(), linewidth=1.5,
            color=color, solid_capstyle="round"
        )

    # Linha de referência em 0%
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="-")

    ax.set_title("Drawdown por Estratégia", loc="left", pad=15)
    ax.set_xlabel("Data", labelpad=10)
    ax.set_ylabel("Drawdown (%)", labelpad=10)
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
    )

    _format_percent_axis(ax, axis="y")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    fig.autofmt_xdate(rotation=30)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[SALVO] {save_path}")
    plt.close()


def plot_allocation_heatmap(
    weights_df: pd.DataFrame,
    title: str = "Alocação Semanal",
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap de alocação por data de rebalanceamento de um cenário.
    """
    label_step = max(1, int(np.ceil(len(weights_df.index) / 24)))
    tick_labels = [
        date.strftime("%d/%m/%Y") if position % label_step == 0 else ""
        for position, date in enumerate(weights_df.index)
    ]

    # Tamanho dinâmico conforme número de datas
    n_dates = len(weights_df.index)
    fig_width = min(24, max(12, n_dates * 0.15))
    n_assets = len(weights_df.columns)
    fig_height = max(5, n_assets * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Decide se mostra anotações (valores) conforme tamanho da matriz
    show_annot = n_dates <= 24 and n_assets <= 12

    sns.heatmap(
        weights_df.T * 100,
        cmap="Blues",
        annot=show_annot,
        fmt=".1f",
        annot_kws={"size": 10, "color": "#111111"},
        cbar_kws={
            "label": "Peso (%)",
            "shrink": 0.6,
            "pad": 0.02,
        },
        linewidths=0.3,
        linecolor="#E0E0E0",
        ax=ax,
        xticklabels=tick_labels,
        yticklabels=True,
    )

    ax.set_title(title, loc="left", pad=15, fontweight="bold")
    ax.set_xlabel("Data de Rebalanceamento", labelpad=10)
    ax.set_ylabel("Ativo", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[SALVO] {save_path}")
    plt.close()


def plot_efficient_frontier(
    daily_returns: pd.DataFrame,
    weights_dict: Dict[str, pd.DataFrame],
    n_random: int = 5000,
    save_path: Optional[Path] = None,
) -> None:
    """Plota o conjunto viável e a fronteira eficiente da última janela."""
    last_rebalance = max(weights.index.max() for weights in weights_dict.values())
    constituent_history = load_constituent_history()
    estimation_window, _, eligible = prepare_estimation_window(
        daily_returns,
        last_rebalance,
        constituent_history,
    )
    if estimation_window.shape[1] < 2:
        raise ValueError("Dados insuficientes para plotar o conjunto viável.")

    expected_returns, covariance, _ = estimate_inputs(estimation_window)
    mean_returns = expected_returns.to_numpy()
    cov_matrix = covariance.to_numpy()
    n_assets = len(mean_returns)
    assets = eligible
    rng = np.random.default_rng(42)
    candidates = rng.dirichlet(np.ones(n_assets), n_random * 4)
    valid = candidates[
        (candidates.min(axis=1) >= CONFIG["optimization"]["min_weight"] - 1e-12)
        & (candidates.max(axis=1) <= CONFIG["optimization"]["max_weight"] + 1e-12)
    ]
    if len(valid) < n_random:
        raise ValueError("Não foi possível simular carteiras suficientes com os limites atuais.")
    random_weights = valid[:n_random]
    rets = random_weights @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, cov_matrix, random_weights))

    bounds = (
        CONFIG["optimization"]["min_weight"],
        CONFIG["optimization"]["max_weight"],
    )
    minimum_frontier = EfficientFrontier(
        expected_returns, covariance, weight_bounds=bounds
    )
    minimum_frontier.min_volatility()
    minimum_return, minimum_volatility, _ = minimum_frontier.portfolio_performance()
    maximum_asset_volatility = float(np.sqrt(np.diag(cov_matrix)).max())
    maximum_return_frontier = EfficientFrontier(
        expected_returns, covariance, weight_bounds=bounds
    )
    maximum_return_frontier.efficient_risk(maximum_asset_volatility * 1.001)
    maximum_return_volatility = maximum_return_frontier.portfolio_performance()[1]

    frontier_volatilities = []
    frontier_returns = []
    if maximum_return_volatility > minimum_volatility + 1e-8:
        targets = np.linspace(
            minimum_volatility * (1 + 1e-5),
            maximum_return_volatility * (1 + 1e-5),
            80,
        )
        for target in targets:
            efficient = EfficientFrontier(
                expected_returns, covariance, weight_bounds=bounds
            )
            efficient.efficient_risk(float(target))
            portfolio_return, portfolio_volatility, _ = efficient.portfolio_performance()
            frontier_returns.append(portfolio_return)
            frontier_volatilities.append(portfolio_volatility)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Nuvem de portfólios aleatórios — sutil
    ax.scatter(
        vols, rets,
        color="#4C78A8",
        marker="o",
        s=8,
        alpha=0.25,
        edgecolors="none",
        label="Carteiras simuladas",
    )
    if frontier_volatilities:
        ax.plot(
            frontier_volatilities,
            frontier_returns,
            color="#111111",
            linewidth=2.5,
            label="Fronteira eficiente",
            zorder=4,
        )
    else:
        ax.scatter(
            minimum_volatility,
            minimum_return,
            color="#111111",
            marker="*",
            s=260,
            label="Fronteira eficiente (ponto único)",
            zorder=7,
        )

    # Os limites verticais representam exatamente a restrição de cada perfil.
    for name, profile in CONFIG["profiles"].items():
        target = profile["target_volatility"]
        color = PALETTE.get(name, "#333333")
        ax.axvline(target, color=color, linestyle="--", linewidth=1.0, alpha=0.6)

    # Carteiras da última data. Deslocamentos evitam sobrepor os rótulos quando
    # diferentes limites de risco resultam na mesma carteira.
    annotation_offsets = {}
    for name, wdf in weights_dict.items():
        color = PALETTE.get(name, "#333333")
        w = wdf.loc[last_rebalance].reindex(assets).fillna(0).values
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        p_ret = np.dot(w, mean_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        marker = "o" if name == "benchmark" else "D"
        size = 150 if name == "benchmark" else 220
        ax.scatter(
            p_vol, p_ret,
            color=color,
            s=size,
            zorder=5,
            edgecolors="white",
            linewidths=2.0,
            marker=marker,
        )
        coordinate = (round(float(p_vol), 6), round(float(p_ret), 6))
        offset_number = annotation_offsets.get(coordinate, 0)
        annotation_offsets[coordinate] = offset_number + 1
        label = (
            "Benchmark 1/N"
            if name == "benchmark"
            else CONFIG["profiles"][name]["name"]
        )
        ax.annotate(
            label,
            (p_vol, p_ret),
            textcoords="offset points",
            xytext=(12, 5 - 18 * offset_number),
            fontsize=13,
            fontweight="bold",
            color=color,
        )

    ax.set_title(
        "Fronteira Eficiente e Conjunto Viável — Última Janela",
        loc="left",
        pad=15,
    )
    ax.set_xlabel("Volatilidade Anualizada", labelpad=10)
    ax.set_ylabel("Retorno Esperado Anualizado", labelpad=10)
    ax.legend(loc="best", frameon=True)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[SALVO] {save_path}")
    plt.close()


def generate_all_figures(
    backtest_results: Dict[str, pd.DataFrame],
    weights_dict: Dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    outdir: Path,
) -> None:
    """Gera todas as figuras do relatório."""
    print("\n" + "=" * 60)
    print("FASE 5 — VISUALIZAÇÃO")
    print("=" * 60)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PUBLICATION_STYLE):
        plot_portfolio_evolution(
            backtest_results,
            save_path=outdir / "evolucao_patrimonio.png",
        )
        plot_drawdowns(
            backtest_results,
            initial_capital=CONFIG["backtest"]["initial_capital"],
            save_path=outdir / "drawdowns.png",
        )
        plot_efficient_frontier(
            daily_returns,
            weights_dict,
            save_path=outdir / "conjunto_viavel.png",
        )
        for name, wdf in weights_dict.items():
            plot_allocation_heatmap(
                wdf,
                title=f"Alocação Semanal — {name.capitalize()}",
                save_path=outdir / f"heatmap_{name}.png",
            )

    print(f"\n[TODAS AS FIGURAS SALVAS EM] {outdir}")
