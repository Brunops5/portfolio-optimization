# Otimização de Portfólios de Criptoativos

Projeto de TCC para avaliar carteiras de criptoativos construídas pela fronteira
eficiente de Markowitz e orientadas a perfis de risco. Em cada rebalanceamento,
o modelo maximiza o retorno esperado respeitando um limite de volatilidade anual.

A explicação completa, passo a passo, está em
[`DOCUMENTACAO_PIPELINE.md`](DOCUMENTACAO_PIPELINE.md).

## Modelo

Para cada cenário de risco, resolve-se o problema clássico de média-variância:

```text
maximizar       w' μ
sujeito a       w' Σ w <= volatilidade_alvo²
                soma(w) = 1
                0 <= w <= 1
```

A implementação usa `PyPortfolioOpt.EfficientFrontier.efficient_risk`. A estimação
de μ e Σ usa as funções padrão da biblioteca (`expected_returns.mean_historical_return`
e `risk_models.sample_cov`); apenas a anualização fixa `frequency=365`, convenção
do projeto para mercados que operam continuamente. A covariância estimada passa
então por `risk_models.fix_nonpositive_semidefinite(..., fix_method="spectral")`,
garantia numérica de que a matriz é PSD antes da otimização.

Os cenários experimentais configurados são três pontos ao longo da região
factível das janelas de estimação:

| Cenário | Volatilidade anual máxima estimada |
|---|---:|
| Risco 65% | 65% |
| Risco 85% | 85% |
| Risco 105% | 105% |

O limite inferior de 65% foi definido a partir da maior volatilidade da carteira
de mínima variância observada nas janelas de estimação, arredondada para cima, de
modo que o menor alvo seja factível em todas as janelas avaliadas. O ponto
intermediário de 85% e o limite superior de 105% preservam, respectivamente, uma
posição de risco médio e o extremo de maior risco do desenho original; em algumas
janelas o teto não é vinculante porque a carteira de máximo retorno já possui
volatilidade inferior. Esse caso é parte do resultado e não deve ser removido. Os
valores devem ser congelados antes da execução oficial do experimento.

Os cenários recebem nomes neutros, em vez de rótulos como conservador, moderado e
agressivo, pois a literatura de média-variância não estabelece cortes universais
de volatilidade para perfis de investidor. Se um alvo for inferior à mínima
volatilidade possível em uma janela, o pipeline usa a carteira de mínima
volatilidade e registra `target_feasible=false`; a violação nunca é ocultada.

O benchmark é uma carteira 1/N, rebalanceada nas mesmas datas e sujeita aos mesmos
custos. O universo não é uma lista fixa: cada data usa somente os constituintes do
NCI cuja reconstituição trimestral já estava efetiva. O histórico auditável está em
[`nci_constituents.yaml`](nci_constituents.yaml).

## Dados e cache

Coinbase é a fonte principal e Yahoo Finance é o fallback somente quando não há
nenhum candle da Coinbase para o produto. Uma série parcial da Coinbase é mantida
como tal — datas anteriores ao lançamento não são preenchidas com um ticker
homônimo do Yahoo. Os dados de mercado
ficam, por padrão, fora do repositório:

```text
~/.cache/portfolio-optimization/market-data/
```

O diretório pode ser alterado pela variável `PORTFOLIO_CACHE_DIR`. O cache guarda
a fonte de cada observação, valida o intervalo disponível e baixa somente trechos
ausentes. Arquivos antigos em `data/raw` são importados automaticamente na primeira
execução. A origem desses arquivos é recuperada do esquema gravado pelo carregador
anterior e marcada como `inferred_from_legacy_schema`; formatos ambíguos são
rejeitados. Uma tentativa de download que falha não é tratada como cobertura e volta
a ser tentada na execução seguinte. Cada execução grava
`data/processed/nci_data_provenance.csv`, com fonte, cobertura e SHA-256 dos dados
efetivamente usados.

O pré-processamento não usa `backward fill`. Somente lacunas internas de até sete
dias são preenchidas para frente; lacunas maiores e datas anteriores à primeira ou
posteriores à última cotação permanecem ausentes. O dataset persistido contém preços
tratados e retornos simples, usados pela otimização e pelo backtest. Os testes
estatísticos convertem os retornos periódicos realizados das carteiras em log-retornos
no momento da comparação.

## Backtest

Os pesos são estimados com os 12 meses anteriores e entram em vigor no primeiro
dia observado de cada semana de segunda a domingo. Entre rebalanceamentos, as
posições sofrem drift conforme os preços variam. O custo de transação é aplicado
uma única vez sobre o turnover.

As comparações estatísticas usam log-retornos semanais não sobrepostos como análise
principal e log-retornos mensais como robustez. A frequência semanal aumenta o
número de pares observados, mas não elimina possível dependência temporal.

Quando uma posição mantida deixa temporariamente de ter cotação, seu último valor é
preservado até o próximo rebalanceamento e o evento é contado na coluna
`stale_positions`. Por padrão, qualquer ocorrência reprova a validação final para
impedir que uma liquidação implícita ao preço antigo seja aceita silenciosamente.

## Instalação e execução

```bash
make install
make run
```

Para ignorar o conteúdo do cache no intervalo solicitado:

```bash
python main.py --force-download
```

## Testes

```bash
make test
make lint
make check
```

A suíte cobre, entre outros pontos, PSD da covariância, limite de volatilidade,
alvos inviáveis, monotonicidade do retorno ótimo com o orçamento de risco, ausência
de look-ahead, cache, drift de pesos e cobrança exata do custo de transação.

## Estrutura

```text
src/data_loader.py    coleta, cache e retornos
src/optimizer.py      universo histórico, estimação e otimização rolling
src/backtester.py     simulação e métricas
src/stats_tests.py    comparações pareadas
src/visualization.py  gráficos
src/validators.py     verificações de consistência
main.py               orquestração do pipeline
config.yaml           parâmetros do experimento
Makefile              comandos de instalação, execução, lint e testes
```

Os resultados são gravados em `results/` e as figuras e tabelas em `reports/`.
