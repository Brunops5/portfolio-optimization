# PortPerfilCrypto

PortPerfilCrypto is an undergraduate research project that builds and evaluates
cryptocurrency portfolios for three experimental volatility limits. It uses
Markowitz mean-variance optimization, historical Nasdaq Crypto Index (NCI)
constituents, weekly rebalancing, and an equally weighted 1/N benchmark.

The code, comments, docstrings, reports, and configuration are in English.
Implementation details are documented next to each function in `src/`.

## Research design

For each risk scenario and rebalance date, the optimizer solves:

```text
maximize       w' μ
subject to     w' Σ w <= target_volatility²
               sum(w) = 1
               0 <= w <= 1
```

Here, `μ` is the vector of expected annual returns, `Σ` is the annualized
covariance matrix, and `w` contains the portfolio weights. The objective is
implemented with `PyPortfolioOpt.EfficientFrontier.efficient_risk`.

The three scenarios are experimental points on the efficient frontier, not
universal investor suitability categories:

| Scenario | Maximum estimated annual volatility |
|---|---:|
| Conservative (65%) | 65% |
| Moderate (85%) | 85% |
| Aggressive (105%) | 105% |

The 65% target is above the minimum feasible volatility observed in the rolling
windows. The other targets represent intermediate and higher-risk points. A
target is a ceiling, not a requirement to use all available risk. If a target is
infeasible, the optimizer returns the minimum-volatility portfolio and records
`target_feasible=false`; it never hides the violation.

## Pipeline

`main.py` runs the following steps in order:

1. **Collect data.** Download daily USD prices from Coinbase Exchange. Yahoo
   Finance is used only when Coinbase returns no candles for the entire product.
2. **Clean data.** Align dates, fill only internal gaps of at most seven days,
   and calculate simple daily returns. Backward fill is never used.
3. **Select the historical universe.** For each rebalance date, select the NCI
   composition already in effect. Assets need at least 300 observations and a
   price no more than seven days old.
4. **Estimate parameters.** Use PyPortfolioOpt's historical mean and sample
   covariance with 365-day annualization. Spectral correction handles numerical
   non-positive-semidefinite covariance matrices.
5. **Optimize portfolios.** Use only the previous 12 months. The rebalance date
   itself is excluded from estimation to prevent look-ahead bias.
6. **Build the benchmark.** Assign weight `1/N` to the same eligible assets on
   the same dates.
7. **Backtest.** Start with US$ 10,000, rebalance weekly, let weights drift
   between rebalances, and charge 0.1% times turnover once per rebalance. The
   initial allocation has no entry cost, consistently across all strategies.
8. **Compare performance.** Calculate total and annualized return, annualized
   volatility, maximum drawdown, turnover, transaction costs, and rebalance
   events.
9. **Run statistical tests.** Compound daily returns into non-overlapping
   weekly log returns for the primary Wilcoxon signed-rank tests. Monthly log
   returns provide a robustness check. Holm-adjusted p-values account for
   multiple comparisons. Partial weeks are excluded.
10. **Validate and publish.** Save final artifacts only if weights, historical
    constituents, covariance matrices, risk targets, backtests, and metrics pass
    validation.

A held asset without a new price keeps its last monetary value until the next
rebalance, and the event is counted in `stale_positions`. By default, any such
event fails final validation because it requires an explicit methodological
treatment.

## Efficient frontier figure

`reports/figures/efficient_frontier.png` uses the last estimation window. The
blue points are randomly simulated feasible portfolios. The black curve is not
an interpolation of those points: it is calculated by solving
`efficient_risk` repeatedly from the minimum-volatility portfolio to the
maximum-return portfolio. Vertical dashed lines show the three risk ceilings.

If the minimum-volatility and maximum-return portfolios coincide, the efficient
set is a single point rather than a curve. This occurs in the current last
window: all estimated expected returns are negative, and the minimum-volatility
portfolio has the highest (least negative) expected return. The three optimized
portfolios therefore coincide in this figure.

Identical risk-return portfolios are combined into one point with a combined
legend label. Their coordinates are never shifted merely to improve readability.
Portfolio evolution and drawdown figures use lines only, without filled areas.

## Data, cache, and provenance

Market data is cached outside the repository by default:

```text
~/.cache/portfolio-optimization/market-data/
```

Set `PORTFOLIO_CACHE_DIR` to use another directory. The loader reuses existing
candles, downloads missing intervals, removes duplicates, and writes cache files
atomically. Failed downloads are not recorded as successful coverage and are
retried on the next run. Legacy files in `data/raw` can be imported when their
original schema identifies the source unambiguously.

Coinbase remains the fixed source after it has recognized a product. In
particular, a shorter Coinbase history is not replaced by a potentially unrelated
Yahoo ticker with the same symbol. Each run writes
`data/processed/nci_data_provenance.csv` with source, coverage, observation count,
missing days, and a SHA-256 checksum.

The historical NCI compositions and their effective dates are recorded in
`nci_constituents.yaml`. The union in `config.yaml` is used for collection only;
optimization uses the composition effective on each rebalance date.

## Installation and execution

Python 3.12 is recommended.

```bash
make install
make run
```

To download the requested period again instead of using cached data:

```bash
python main.py --force-download
```

Run the automated checks with:

```bash
make test
make lint
make check
```

## Configuration

All experimental parameters are in `config.yaml`:

- analysis period: 2022-01-01 to 2025-12-31;
- estimation window: 12 months;
- rebalance frequency: weekly, Monday to Sunday;
- positions: long-only and fully invested;
- initial capital: US$ 10,000;
- transaction cost: 0.1% times turnover;
- annualization: 365 days;
- primary statistical frequency: weekly;
- robustness statistical frequency: monthly.

Shared configuration keys, data column names, strategy identifiers, and output
names are centralized in `src/names.py`. Experimental parameters should be fixed
before the official run and should not be tuned to obtain a preferred result.

## Project structure

```text
main.py                  Pipeline orchestration
config.yaml              Experiment parameters
nci_constituents.yaml    Point-in-time NCI compositions
src/names.py             Shared constants
src/data_loader.py       Data collection, cache, and returns
src/optimizer.py         Rolling estimation and optimization
src/backtester.py        Backtest and performance metrics
src/stats_tests.py       Paired statistical comparisons
src/validators.py        Mathematical and financial checks
src/visualization.py     Figures
tests/                   Automated tests
Makefile                 Installation and validation commands
```

## Outputs

```text
data/processed/nci_prices.csv
data/processed/nci_daily_simple.csv
data/processed/nci_data_provenance.csv
results/weights/weights_<strategy>.csv
results/optimization/optimization_diagnostics.csv
results/metrics/backtest_<strategy>.csv
results/metrics/metrics_summary.csv
reports/tables/wilcoxon_weekly.csv
reports/tables/wilcoxon_weekly.md
reports/tables/wilcoxon_monthly_robustness.csv
reports/tables/wilcoxon_monthly_robustness.md
reports/figures/portfolio_evolution.png
reports/figures/drawdowns.png
reports/figures/efficient_frontier.png
reports/figures/heatmap_<strategy>.png
```

## Current experiment summary

The configured execution produced the following out-of-sample results:

| Strategy | Total return | Annualized return | Realized volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Conservative (65%) | 221.13% | 47.53% | 59.58% | -37.84% |
| Moderate (85%) | 113.00% | 28.67% | 73.27% | -43.23% |
| Aggressive (105%) | 91.71% | 24.23% | 76.38% | -45.78% |
| Benchmark 1/N | 126.31% | 31.29% | 65.15% | -54.01% |

A higher cumulative return does not by itself establish statistical superiority.
Interpret these results together with risk, drawdown, costs, concentration,
weekly and monthly Wilcoxon tests, and Holm-adjusted p-values. The historical
mean is especially unstable for cryptocurrencies, and pure return maximization
can produce concentrated portfolios. Cash or a risk-free asset is intentionally
excluded; including either would change the research universe.

## Reproducibility

Preserve the configuration, dependency versions, data provenance, optimization
diagnostics, generated weights and results, and the corresponding code commit.
The scientific objective is to report favorable and unfavorable out-of-sample
results, not to adjust the algorithm until it beats the benchmark.
