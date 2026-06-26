# Otimização de Portfólios de Criptoativos

> **Trabalho de Conclusão de Curso** — Bacharelado em Engenharia de Computação  
> **Autor:** Bruno Augusto de Oliveira  
> **Orientador:** Prof. Dr. Marcos Roberto Ribeiro  
> **Instituição:** IFMG — Campus Bambuí

---

## Resumo

Este projeto implementa um sistema de **recomendação de portfólios de criptoativos orientada a perfis de risco**, com validação histórica comparativa (backtesting). O universo de ativos compreende os componentes do **Nasdaq Crypto Index (NCI)**, e a otimização é conduzida mensalmente por meio de **janelas deslizantes de 12 meses**.

Três perfis de investidor são definidos por meio de uma **restrição customizada de correlação média máxima**:

| Perfil | Correlação Média Máxima |
|--------|------------------------|
| Conservador | ≤ 0,50 |
| Moderado | ≤ 0,70 |
| Agressivo | ≤ 0,85 |

O desempenho das estratégias otimizadas é comparado contra um **benchmark de alocação igualitária (1/N)** e avaliado estatisticamente via **teste de Wilcoxon de postos sinalizados**.

---

## Tecnologias

- **Python 3.12**
- **PyPortfolioOpt** — framework base para otimização mean-variance (Markowitz)
- **scipy.optimize** — otimização não-linear com restrição customizada de correlação
- **pandas / numpy** — manipulação e computação numérica
- **matplotlib / seaborn** — visualização
- **Coinbase Exchange API** + **Yahoo Finance** (fallback) — fonte de dados

---

## Estrutura do Projeto

```
optimization/
├── src/
│   ├── data_loader.py       # Coleta e pré-processamento de dados
│   ├── optimizer.py         # Motor de otimização (PyPortfolioOpt + restrição customizada)
│   ├── backtester.py        # Simulação histórica e métricas de desempenho
│   ├── stats_tests.py       # Teste de Wilcoxon e estatísticas descritivas
│   ├── visualization.py     # Geração de gráficos e relatórios
│   └── validators.py        # Sanity checks e validação do experimento
├── tests/                   # Testes unitários com pytest
│   ├── test_optimizer.py
│   ├── test_backtester.py
│   ├── test_stats_tests.py
│   ├── test_data_loader.py
│   └── test_validators.py
├── data/
│   ├── raw/                 # Dados OHLCV brutos da Coinbase
│   └── processed/           # Retornos logarítmicos diários/mensais
├── results/
│   ├── weights/             # Pesos mensais otimizados por perfil
│   └── metrics/             # Métricas de backtest
├── reports/
│   ├── figures/             # Gráficos gerados
│   └── tables/              # Tabelas de resultados estatísticos
├── config.yaml              # Parâmetros do projeto
├── main.py                  # Pipeline principal (orquestra todas as fases)
├── requirements.txt         # Dependências Python
├── requirements-dev.txt     # Dependências de desenvolvimento
├── Dockerfile               # Containerização
├── docker-compose.yml       # Orquestração Docker
├── .dockerignore            # Exclusões do build Docker
├── .gitignore               # Exclusões do versionamento Git
├── PLAN.md                  # Plano de execução detalhado
└── README.md                # Documentação do projeto
```

---

## Instalação

### Opção 1: Ambiente Virtual (venv)

```bash
# Clone o repositório (se aplicável)
cd optimization

# Cria e ativa o ambiente virtual
python3.12 -m venv venv
source venv/bin/activate

# Instala dependências
pip install -r requirements.txt
```

### Opção 2: Docker

```bash
cd optimization

# Build da imagem
docker compose build

# Execução do pipeline
docker compose up
```

Os dados e resultados são persistidos via **bind mounts** nos diretórios `data/`, `results/` e `reports/`.

---

## Execução

### Pipeline completo

```bash
# Usando ambiente virtual
source venv/bin/activate
python main.py

# Usando Docker
docker compose up
```

### Forçar re-download dos dados

```bash
python main.py --force-download
```

### Apenas um módulo específico

```bash
# Testar otimizador com dados dummy
python src/optimizer.py

# Testar backtester
python src/backtester.py

# Testar estatísticas
python src/stats_tests.py
```

---

## Testes Unitários

O projeto possui uma suite de testes com **pytest**. Para executar:

```bash
# Instala dependências de desenvolvimento
pip install -r requirements-dev.txt

# Executa todos os testes
pytest

# Com cobertura de código
pytest --cov=src --cov-report=term-missing
```

### Arquivos de teste

| Teste | Cobertura |
|---|---|
| `tests/test_optimizer.py` | Correlação média ponderada e otimização de portfólio |
| `tests/test_backtester.py` | Backtest, drawdown e métricas de desempenho |
| `tests/test_stats_tests.py` | Retornos mensais logarítmicos e teste de Wilcoxon |
| `tests/test_data_loader.py` | Cálculo de retornos logarítmicos e preenchimento de dados |

---

## Configuração

Os parâmetros do projeto são centralizados em **`config.yaml`**:

```yaml
# Período de análise
start_date: "2022-01-01"
end_date: "2025-12-31"

# Janela de estimação (meses)
estimation_window_months: 12

# Ativos (componentes do NCI na Coinbase)
assets: [BTC, ETH, SOL, XRP, ADA, AVAX, MATIC, LINK, UNI, LTC]

# Perfis de risco
profiles:
  conservador: { max_avg_correlation: 0.50 }
  moderado:    { max_avg_correlation: 0.70 }
  agressivo:   { max_avg_correlation: 0.85 }

# Parâmetros de otimização
optimization:
  risk_free_rate_annual: 0.045 
  max_weight: 0.40               # Limite de concentração por ativo
  objective: "max_sharpe"

# Parâmetros de backtest
backtest:
  initial_capital: 10000.0
  transaction_cost: 0.001        # 0.1% por rebalanceamento
```

---

## Fases do Pipeline

O `main.py` orquestra 6 fases sequenciais:

1. **Coleta de Dados** — Download OHLCV via Coinbase API (com fallback Yahoo Finance)
2. **Otimização** — Janela deslizante de 12 meses, maximizando Sharpe com restrição de correlação
3. **Backtesting** — Simulação mensal com custos de transação e cálculo de métricas
4. **Análise Estatística** — Teste de Wilcoxon pareado entre estratégias
5. **Visualização** — Gráficos de evolução, drawdown, fronteira eficiente e heatmaps
6. **Validação** — Sanity checks matemáticos e financeiros automatizados

---

## Validação dos Resultados

O projeto inclui um módulo de validação (`src/validators.py`) que verifica automaticamente a coerência do experimento. O relatório de validação é exibido ao final da execução do `main.py`.

### Verificações realizadas

| Categoria | O que valida |
|---|---|
| **Pesos Otimizados** | Soma = 1, não-negatividade, limite de 40% por ativo, carteira não vazia |
| **Restrição de Correlação** | Correlação média dentro de [-1, 1]; avisa quando fallback ocorre |
| **Consistência do Backtest** | Patrimônio positivo, retornos diários realistas, consistência P_t = P_{t-1}·e^{r_t}, drawdown ≤ -100% |
| **Métricas de Desempenho** | Retorno total em range realista, volatilidade > 0, Sharpe/Calmar finitos |

### Como interpretar

- ✅ **Aprovado**: todas as verificações passaram.
- ⚠️ **Aviso**: situação esperada do domínio (ex.: fallback da restrição de correlação quando o mercado é intrinsecamente mais correlacionado que o limite do perfil).
- ❌ **Erro**: problema que invalida o experimento (ex.: soma dos pesos diferente de 1).

### Validação manual dos gráficos

Para garantir que os gráficos estão corretos, verifique:

1. **Evolução do patrimônio**: o benchmark (1/N) e as estratégias otimizadas partem de $10.000?
2. **Drawdown**: os valores estão entre 0% e -100%? A ordem dos drawdowns faz sentido (mais agressivo = drawdown maior)?
3. **Fronteira eficiente**: os perfis otimizados estão acima/na fronteira dos portfólios aleatórios?
4. **Heatmaps**: a soma dos pesos em cada coluna é 100%? Os ativos com peso zero aparecem em branco?

---

## Resultados Gerados

Após a execução, os seguintes artefatos são produzidos:

| Artefato | Localização | Descrição |
|----------|-------------|-----------|
| Pesos otimizados | `results/weights/weights_*.csv` | Alocação mensal por perfil |
| Métricas de backtest | `results/metrics/backtest_*.csv` | Série temporal de patrimônio |
| Resumo de métricas | `results/metrics/metrics_summary.csv` | Tabela comparativa |
| Teste de Wilcoxon | `reports/tables/wilcoxon_results.md` | Resultados estatísticos |
| Evolução do patrimônio | `reports/figures/evolucao_patrimonio.png` | Gráfico log-scale |
| Drawdowns | `reports/figures/drawdowns.png` | Máxima queda por estratégia |
| Fronteira eficiente | `reports/figures/fronteira_eficiente.png` | Simulação Monte Carlo |
| Heatmaps de alocação | `reports/figures/heatmap_*.png` | Pesos mensais por ativo |

---

## Metodologia de Otimização

O sistema utiliza uma arquitetura híbrida:

1. **PyPortfolioOpt** calcula os inputs acadêmicos:
   - `expected_returns.mean_historical_return()` — retornos esperados anualizados
   - `risk_models.sample_cov()` — matriz de covariância anualizada

2. **scipy.optimize.minimize** (SLSQP) executa a otimização propriamente dita,
   incorporando a **restrição customizada de correlação média máxima**:

   ```
   ρ̄(w) = Σ_i Σ_{j>i} w_i w_j ρ_{ij} / Σ_i Σ_{j>i} w_i w_j ≤ limite
   ```

   Como essa restrição é não-convexa, ela não pode ser expressa diretamente no
   solver `cvxpy` do PyPortfolioOpt. A abordagem híbrida permite manter a
   fidelidade ao framework acadêmico (PyPortfolioOpt) enquanto estende o modelo
   com a restrição de correlação proposta no artigo.

---

## Métricas de Desempenho

| Métrica | Fórmula / Descrição |
|---------|---------------------|
| Retorno Total | (Valor final / Valor inicial) − 1 |
| Retorno Anualizado | (1 + Retorno Total)^(1/n) − 1 |
| Volatilidade Anualizada | σ_diário × √252 |
| Sharpe Ratio | (Retorno Anualizado − Rf) / Volatilidade |
| Máximo Drawdown | Mínimo queda do pico ao vale |
| Calmar Ratio | Retorno Anualizado / \|MDD\| |

---

## Licença

Este projeto é de uso acadêmico. As bibliotecas de terceiros utilizadas são
software livre (MIT, BSD, Apache 2.0).

---