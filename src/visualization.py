"""
Módulo de visualização e geração de relatórios gráficos.
Estilo orientado a publicação acadêmica — limpo, legível e profissional.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO GLOBAL DE ESTILO
# ---------------------------------------------------------------------------
REPORTS_FIGS = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORTS_FIGS.mkdir(parents=True, exist_ok=True)

# Paleta institucional — cores sóbrias e distinguíveis
PALETTE = {
    "conservador": "#1B5E20",   # verde escuro
    "moderado": "#1565C0",      # azul institucional
    "agressivo": "#C62828",     # vermelho sóbrio
    "benchmark": "#616161",     # cinza médio
}

# Configurações matplotlib para publicação
plt.rcParams.update(
    {
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
)


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
    initial_capital: float = 10000.0,
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
    ax.set_ylim(ymin - margin, ymax + margin)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[SALVO] {save_path}")
    plt.close()


def plot_drawdowns(
    backtest_results: Dict[str, pd.DataFrame],
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
        rolling_max = pv.cummax()
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
    title: str = "Alocação Mensal",
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap de alocação mensal de um perfil.
    Estilo: cmap 'Blues', rótulos de datas MM/YYYY, linhas sutis.
    """
    # Formata datas como MM/YYYY
    tick_labels = [d.strftime("%m/%Y") for d in weights_df.index]

    # Tamanho dinâmico conforme número de datas
    n_dates = len(weights_df.index)
    fig_width = max(12, n_dates * 0.45)
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
    """
    Simula portfólios aleatórios e plota a fronteira eficiente com os perfis.
    Estilo: pontos sutis, perfis destacados com marcadores grandes e bordas.
    """
    mean_returns = daily_returns.mean().values
    cov_matrix = daily_returns.cov().values
    n_assets = len(mean_returns)
    assets = daily_returns.columns.tolist()

    np.random.seed(42)
    random_weights = np.random.dirichlet(np.ones(n_assets), n_random)
    rets = random_weights @ mean_returns * 365
    vols = np.sqrt(np.array([w @ cov_matrix @ w for w in random_weights])) * np.sqrt(365)
    sharpe_random = np.divide(rets, vols, out=np.zeros_like(rets), where=vols > 1e-12)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Nuvem de portfólios aleatórios — sutil
    scatter = ax.scatter(
        vols, rets,
        c=sharpe_random,
        cmap="YlGnBu",
        marker="o",
        s=8,
        alpha=0.35,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Sharpe Ratio", rotation=270, labelpad=18, fontsize=13)

    # Linhas iso-Sharpe de referência
    for s in [0.5, 1.0, 1.5, 2.0]:
        vol_line = np.linspace(vols.min() * 0.8, vols.max() * 1.1, 200)
        ret_line = s * vol_line
        ax.plot(vol_line, ret_line, "k--", alpha=0.15, linewidth=0.8)
        # Anotação pequena da linha
        ax.text(
            vol_line[-1], ret_line[-1], f"  s={s:.1f}",
            fontsize=9, color="#555555", va="center"
        )

    # Perfis otimizados (último rebalanceamento)
    for name, wdf in weights_dict.items():
        if name == "benchmark":
            continue
        color = PALETTE.get(name, "#333333")
        w = wdf.iloc[-1].reindex(assets).fillna(0).values
        p_ret = np.dot(w, mean_returns) * 365
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(365)

        ax.scatter(
            p_vol, p_ret,
            color=color,
            s=220,
            zorder=5,
            edgecolors="white",
            linewidths=2.0,
            marker="D",  # diamante
        )
        ax.annotate(
            name.capitalize(),
            (p_vol, p_ret),
            textcoords="offset points",
            xytext=(12, 5),
            fontsize=13,
            fontweight="bold",
            color=color,
        )

    ax.set_title("Fronteira Eficiente Simulada — Modelo de Markowitz", loc="left", pad=15)
    ax.set_xlabel("Volatilidade Anualizada", labelpad=10)
    ax.set_ylabel("Retorno Esperado Anualizado", labelpad=10)

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
    outdir: Path = REPORTS_FIGS,
) -> None:
    """Gera todas as figuras do relatório."""
    print("\n" + "=" * 60)
    print("FASE 5 — VISUALIZAÇÃO")
    print("=" * 60)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_portfolio_evolution(
        backtest_results,
        save_path=outdir / "evolucao_patrimonio.png",
    )
    plot_drawdowns(
        backtest_results,
        save_path=outdir / "drawdowns.png",
    )
    plot_efficient_frontier(
        daily_returns,
        weights_dict,
        save_path=outdir / "fronteira_eficiente.png",
    )
    for name, wdf in weights_dict.items():
        plot_allocation_heatmap(
            wdf,
            title=f"Alocação Mensal — {name.capitalize()}",
            save_path=outdir / f"heatmap_{name}.png",
        )

    print(f"\n[TODAS AS FIGURAS SALVAS EM] {outdir}")


if __name__ == "__main__":
    # Teste rápido de estilo
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    bt = pd.DataFrame(
        {
            "portfolio_value": 10000 * np.exp(np.random.normal(0.0008, 0.02, len(dates)).cumsum()),
            "daily_return": np.random.normal(0.0008, 0.02, len(dates)),
        },
        index=dates,
    )
    plot_portfolio_evolution({"teste": bt}, save_path=REPORTS_FIGS / "test.png")
