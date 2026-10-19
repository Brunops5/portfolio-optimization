# Documentação do Pipeline de Otimização de Portfólios de Criptoativos

## 1. Objetivo do projeto

Este projeto constrói e avalia carteiras formadas exclusivamente por criptoativos.
O objetivo é encontrar, para diferentes níveis experimentais de risco, a carteira com maior
retorno esperado possível sem ultrapassar um limite predefinido de risco.

O risco é representado pela volatilidade anual estimada da carteira. Foram definidos
três cenários:

| Cenário | Volatilidade anual máxima estimada |
|---|---:|
| Risco 65% | 65% |
| Risco 85% | 85% |
| Risco 105% | 105% |

Esses percentuais podem parecer elevados quando comparados a ações ou títulos de
renda fixa, mas refletem a volatilidade estruturalmente alta de uma carteira formada
somente por criptoativos. Na amostra inicial, a própria carteira de mínima variância
apresentou volatilidade próxima de 63% em algumas janelas. Portanto, limites muito
inferiores poderiam tornar o problema matematicamente inviável sem permitir caixa
ou um ativo livre de risco.

O desempenho das carteiras é comparado com um benchmark 1/N, no qual o capital é
dividido igualmente entre todos os ativos elegíveis em cada rebalanceamento.

## 2. Visão geral

O ponto de entrada do sistema é `main.py`. O pipeline é composto pelas seguintes
fases:

```text
configuração
    ↓
coleta e cache dos preços
    ↓
limpeza e cálculo dos retornos
    ↓
otimização rolling para cada cenário
    ↓
construção do benchmark 1/N
    ↓
backtest com rebalanceamento semanal
    ↓
métricas e testes estatísticos
    ↓
figuras, tabelas e validações
```

As fases utilizam os seguintes módulos:

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestra todas as fases. |
| `config.yaml` | Centraliza os parâmetros do experimento. |
| `src/data_loader.py` | Coleta, cache, alinhamento de preços e retornos. |
| `src/optimizer.py` | Universo histórico, estimação e otimização. |
| `src/backtester.py` | Simula a aplicação histórica das carteiras. |
| `src/stats_tests.py` | Executa comparações estatísticas. |
| `src/visualization.py` | Gera gráficos do experimento. |
| `src/validators.py` | Verifica a consistência matemática e financeira. |

## 3. Configuração do experimento

Os principais parâmetros estão em `config.yaml`:

- período de dados: janeiro de 2022 a dezembro de 2025;
- universo: composição histórica trimestral do NCI, registrada em
  `nci_constituents.yaml`;
- janela de estimação: 12 meses;
- rebalanceamento: semanal, no primeiro dia observado da semana iniciada na segunda-feira;
- mercado: criptoativos negociados contra dólar;
- posições: somente compradas, sem venda a descoberto;
- capital inicial: US$ 10.000;
- custo de transação: 0,1% sobre o turnover;
- anualização: 365 dias, pois o mercado de criptoativos funciona continuamente;
- objetivo: máximo retorno esperado sob o limite de risco de cada cenário.

O objetivo e a política de manter o valor de uma posição em dias sem cotação são
decisões metodológicas únicas da implementação, não alternativas selecionáveis.
Por isso, não aparecem como opções no YAML. Qualquer posição sem cotação continua
registrada e, por padrão, reprova a validação final.

A grade de risco foi definida com três pontos ao longo da região factível das
janelas rolling. Nas janelas de estimação avaliadas, a volatilidade mínima
permaneceu abaixo de 65%; assim, 65% é o menor alvo fixo factível em todas as
janelas. O ponto de 85% marca uma posição intermediária e o teto de 105%
mantém o extremo superior do desenho inicial.

Esses valores não representam fronteiras universais entre perfis de investidor.
A teoria de média-variância sustenta a comparação de pontos ao longo da fronteira
eficiente, mas a escolha individual depende da tolerância ao risco. Por isso, os
identificadores são neutros (`risco_065` a `risco_105`).

O desenho segue Markowitz (1952) e Merton (1972) ao representar alternativas pela
fronteira eficiente. No contexto de criptoativos, Aljinović, Marasović e Šestanović
(2021) fornecem precedente para comparar posições distintas da fronteira em uma
avaliação fora da amostra; o presente estudo usa mais pontos para observar a
sensibilidade ao teto de volatilidade.

Centralizar os parâmetros evita que valores importantes fiquem espalhados pelo
código e permite reproduzir exatamente uma execução. Para o experimento oficial,
esses parâmetros devem ser definidos antes da análise dos resultados e mantidos
fixos, reduzindo o risco de overfitting.

## 4. Coleta e cache dos dados

### 4.1 Fontes

A fonte principal é a API pública da Coinbase Exchange. Para cada ativo é solicitado
o par `ATIVO-USD`, com candles diários contendo abertura, máxima, mínima,
fechamento e volume.

Quando a Coinbase não retorna nenhum candle para o produto, o Yahoo Finance é usado
como fallback. Cobertura parcial, por si só, não aciona a troca: ela pode indicar
que o ativo ainda não existia no começo do intervalo. Depois que um produto é
identificado na Coinbase, as ampliações do seu cache ficam fixadas nessa fonte. A
fonte efetivamente utilizada é registrada na coluna `source` do arquivo de cache.

Essa hierarquia foi escolhida porque:

1. a Coinbase representa um mercado diretamente negociável de criptoativos;
2. sua API fornece candles públicos sem autenticação;
3. o Yahoo Finance permite obter um ativo quando o par inteiro está indisponível na
   fonte principal;
4. registrar a origem permite auditar a procedência de cada série.

Essa fixação de fonte é importante para símbolos ambíguos. `ARB-USD` no Yahoo pode
designar um ativo anterior ao Arbitrum. A série usada para Arbitrum vem da Coinbase
e começa em 23/03/2023; o pipeline não preenche o período anterior com o homônimo.
Pela mesma regra, a série Coinbase de SAND começa em 26/05/2022.

### 4.2 Divisão das requisições

A Coinbase limita a quantidade de candles retornados. O período solicitado é,
portanto, dividido em blocos de no máximo 300 observações diárias. Os intervalos
são construídos sem sobreposição e os timestamps são deduplicados após a coleta.

Em caso de limite de requisições HTTP 429, o código usa espera exponencial. Isso
reduz a pressão sobre a API e aumenta a chance de a coleta terminar sem intervenção.

### 4.3 Cache persistente

Os dados brutos são armazenados, por padrão, em:

```text
~/.cache/portfolio-optimization/market-data/
```

O local pode ser alterado pela variável de ambiente `PORTFOLIO_CACHE_DIR`. Essa
possibilidade também facilita testes automatizados.

Cada ativo possui um CSV com os candles. A cobertura é sempre inferida das datas
realmente presentes nesse arquivo; uma tentativa que falhou não é registrada como
trecho coberto e será repetida na próxima execução.

Ao executar novamente o pipeline, o carregador:

1. procura o ativo no cache;
2. verifica a menor e a maior data realmente presentes no CSV;
3. reutiliza os dados existentes;
4. baixa somente intervalos ainda não cobertos;
5. combina dados antigos e novos;
6. remove duplicidades;
7. grava o resultado de forma atômica.

A gravação atômica primeiro cria um arquivo temporário e depois o move para o nome
definitivo. Isso reduz o risco de deixar um cache parcialmente escrito caso a
execução seja interrompida.

Arquivos antigos de `data/raw` podem ser importados automaticamente na primeira
execução. O carregador antigo gravava candles Coinbase na ordem
`timestamp, low, high, open, close, volume` e candles Yahoo na ordem
`timestamp, open, high, low, close, volume`. Essa diferença permite recuperar a
fonte, que fica explicitamente marcada como `inferred_from_legacy_schema`; um
esquema ambíguo interrompe a importação. Depois disso, os experimentos deixam de
depender da cópia de dados dentro do repositório.

Para cada execução, `data/processed/nci_data_provenance.csv` registra ativo, produto,
fonte, modo de identificação da fonte, cobertura, quantidade de observações, dias
ausentes e SHA-256 da série efetivamente utilizada.

### 4.4 Caso do MATIC

A série disponível de MATIC termina em 14/10/2025. Nem Coinbase nem Yahoo Finance
retornaram observações posteriores para `MATIC-USD`. O intervalo permanece marcado
como ausente, e uma falha de coleta não é convertida em cobertura permanente.

Isso não afeta mais o backtest: MATIC integrou o NCI em Q2 e Q3 de 2024 e foi
removido na reconstituição efetiva de dezembro de 2024, muito antes do fim de sua
série de preços.

## 5. Preparação dos preços e retornos

### 5.1 Alinhamento temporal

As séries individuais são combinadas em uma matriz na qual:

- cada linha representa um dia;
- cada coluna representa um ativo;
- cada célula contém o preço de fechamento.

Os timestamps são normalizados sem fuso horário para permitir a combinação de
fontes diferentes.

### 5.2 Tratamento de valores ausentes

Somente lacunas internas de até sete dias, localizadas entre a primeira e a última cotação válida de
um ativo, são preenchidas por propagação para frente (`forward fill`).

Não é utilizado `backward fill`. Datas anteriores à primeira cotação permanecem
ausentes, pois preencher esses valores com uma cotação futura introduziria
look-ahead bias. Datas posteriores à última cotação também permanecem ausentes,
evitando criar artificialmente um histórico após o desaparecimento do ativo.

### 5.3 Retornos simples

O retorno simples do ativo `i` no dia `t` é:

```text
r(t,i) = P(t,i) / P(t-1,i) - 1
```

Retornos simples são usados na otimização e no backtest. Eles são adequados para
calcular o retorno de uma carteira em um período:

```text
r_portfólio = soma(w(i) × r(i))
```

### 5.4 Log-retornos

O log-retorno é:

```text
g(t,i) = ln(1 + r(t,i)) = ln(P(t,i) / P(t-1,i))
```

Log-retornos não são combinados diretamente pelos pesos no backtest, pois a média
ponderada de log-retornos não representa exatamente o retorno de uma carteira
autofinanciada. Eles são derivados dos retornos periódicos realizados apenas no
momento da comparação estatística.

São produzidos dois arquivos processados:

- `nci_prices.csv`;
- `nci_daily_simple.csv`;

## 6. Janela móvel e elegibilidade dos ativos

A otimização utiliza uma janela móvel de 12 meses. Para o rebalanceamento de
1º de janeiro de 2024, por exemplo, somente observações de 1º de janeiro a 31 de
dezembro de 2023 podem ser utilizadas.

A condição usada no código é equivalente a:

```text
início_da_janela <= data_da_observação < data_do_rebalanceamento
```

O limite estritamente menor que a data do rebalanceamento garante que nenhum
retorno futuro participa da decisão.

Primeiro, o pipeline consulta `nci_constituents.yaml` e seleciona a composição cuja
data efetiva é a mais recente que não ultrapassa a data do rebalanceamento. Isso
impede usar retroativamente constituintes anunciados no futuro. Dentro desse universo,
um ativo somente é elegível quando:

1. possui pelo menos 300 observações na janela;
2. sua última observação está a no máximo sete dias do rebalanceamento.

Essas condições evitam estimar média e covariância a partir de históricos muito
curtos ou selecionar um ativo que já deixou de ter dados recentes.

Depois da seleção, são mantidas apenas datas completas para o conjunto elegível.
Isso garante que todos os elementos da matriz de covariância sejam estimados sobre
as mesmas observações.

## 7. Estimação dos parâmetros de Markowitz

### 7.1 Retorno esperado

O vetor de retornos esperados é estimado pela função padrão
`expected_returns.mean_historical_return` do PyPortfolioOpt, com compounding
(argumento padrão da biblioteca). A annualização geométrica da média dos
retornos simples diários é:

```text
μ(i) = ( Π_t (1 + r(t,i)) )^(365 / n) - 1
```

Foi mantido o comportamento padrão da biblioteca porque ele é a referência
neutra: qualquer desvio (shrinkage, média exponencial, Black-Litterman)
introduziria uma escolha metodológica adicional a ser justificada. A
anualização usa 365 dias, convenção do projeto para mercados que operam
continuamente. A estimativa é simples e transparente, mas também é sensível ao
período histórico. Por esse motivo, a avaliação fora da amostra é uma parte
importante do experimento.

### 7.2 Matriz de covariância

A matriz de covariância é estimada pela função padrão `risk_models.sample_cov`
do PyPortfolioOpt: a covariância amostral dos retornos diários, anualizada por
365:

```text
Σ_anual = 365 × Cov(retornos_diários)
```

A volatilidade prevista de uma carteira é:

```text
σ_portfólio = raiz(w' Σ w)
```

### 7.3 Correção PSD

`risk_models.sample_cov` já aplica internamente a correção PSD. Para que o
diagnóstico antes/depois seja real, a covariância amostral bruta é calculada
separadamente e a versão usada na otimização é obtida por:

```python
risk_models.sample_cov(
    returns,
    returns_data=True,
    frequency=365,
    fix_method="spectral",
)
```

Uma matriz de covariância deve ser positiva semidefinida para que o solver do
problema de média-variância opere corretamente. Embora a estimação com
observações completas garanta essa propriedade em teoria, ruído numérico pode
produzir pequenos autovalores negativos e causar falhas no solver. O método
espectral substitui autovalores negativos por zero e reconstrói a matriz. O
menor autovalor antes e depois da correção é salvo no diagnóstico de cada
otimização, e a validação final reprova a execução se alguma matriz
permanecer não PSD.

## 8. Otimização por cenário de risco

### 8.1 Formulação

A variável de decisão é o vetor de pesos `w`. A função implementada é:

```text
maximizar       w' μ

sujeito a       w' Σ w <= σ_alvo²
                soma(w) = 1
                0 <= w(i) <= 1
```

A função objetivo maximiza o retorno esperado. A restrição quadrática limita a
volatilidade anual estimada da carteira ao teto de risco do cenário.

### 8.2 PyPortfolioOpt

O problema é resolvido pelo PyPortfolioOpt, com os estimadores padrão da
biblioteca e limites de peso também padrão (long-only, `weight_bounds=(0, 1)`):

```python
mu = expected_returns.mean_historical_return(returns, returns_data=True, frequency=365)
cov = risk_models.sample_cov(returns, returns_data=True, frequency=365)
frontier = EfficientFrontier(mu, cov)
frontier.efficient_risk(target_volatility=target)
```

`efficient_risk()` maximiza o retorno esperado dentro do limite de volatilidade. O
risco resultante pode ser inferior ao alvo quando a carteira de máximo retorno já
se encontra abaixo dele; o limite é um teto, não uma obrigação de consumir risco.

Essa formulação é convexa e usa o backend CVXPY do PyPortfolioOpt. O SciPy não é
mais usado para resolver a alocação; ele permanece no projeto para os testes
estatísticos. Além da anualização de 365 dias, a única intervenção sobre a
estimação é a correção PSD espectral da seção 7.3; os estimadores permanecem
os padrões da biblioteca.

### 8.3 Alvos inviáveis

Antes de resolver cada cenário, o pipeline calcula a carteira de mínima volatilidade.
Se:

```text
σ_alvo < σ_mínima_factível
```

não existe carteira totalmente investida em criptoativos que satisfaça o cenário.
Nesse caso, o sistema:

1. usa a carteira de mínima volatilidade;
2. marca `target_feasible = false`;
3. registra o alvo e o mínimo factível;
4. emite um aviso na validação.

O limite nunca é relaxado silenciosamente. Na execução semanal atual, os 471
problemas — 157 semanas vezes três cenários — foram factíveis.

## 9. Benchmark 1/N

Em cada data, o benchmark atribui:

```text
w(i) = 1 / N
```

a todos os ativos elegíveis naquela janela. Ativos não elegíveis recebem peso zero.

O benchmark:

- usa as mesmas datas de rebalanceamento;
- utiliza a mesma composição histórica do NCI e os mesmos filtros de dados;
- paga os mesmos custos de transação;
- é submetido ao mesmo tratamento de preços ausentes.

Essa equivalência é necessária para que a comparação não favoreça artificialmente
nenhuma estratégia. O 1/N é uma referência importante porque não depende de
estimativas de retornos esperados ou da inversão de matrizes de covariância.

## 10. Backtest

### 10.1 Inicialização

O patrimônio inicial é US$ 10.000. Na primeira data do backtest, o capital é
distribuído conforme os primeiros pesos. Não é cobrado custo de entrada inicial;
essa regra vale igualmente para todos os perfis e para o benchmark.

### 10.2 Drift dos pesos

Entre dois rebalanceamentos, os valores monetários das posições evoluem conforme
os retornos dos ativos:

```text
posição(t,i) = posição(t-1,i) × (1 + r(t,i))
```

O patrimônio é a soma das posições:

```text
V(t) = soma(posição(t,i))
```

Consequentemente, os pesos variam naturalmente durante a semana. Isso representa uma
carteira buy-and-hold entre rebalanceamentos. Manter os pesos constantes diariamente
representaria, incorretamente, rebalanceamento diário.

### 10.3 Rebalanceamento e turnover

No primeiro dia observado de cada semana, os pesos atuais são calculados a partir do valor das
posições antes da negociação. O turnover é:

```text
turnover = soma(|peso_alvo - peso_atual|)
```

O custo monetário é:

```text
custo = patrimônio × taxa_de_transação × turnover
```

O custo é descontado uma única vez. Em seguida, o patrimônio líquido é redistribuído
de acordo com os novos pesos.

### 10.4 Retorno diário líquido

Após custo e variação das posições, o retorno líquido registrado é:

```text
r_portfólio(t) = V(t) / V(t-1) - 1
```

Assim, a própria série de retornos reproduz exatamente a evolução patrimonial.

### 10.5 Posição sem nova cotação

Se um ativo mantido deixa de apresentar retorno, a política atual
`hold_last_value` conserva o último valor monetário conhecido até o próximo
rebalanceamento. Cada ocorrência é registrada em `stale_positions`.

Essa política evita inventar preços futuros durante o pré-processamento. Como uma
posição sem cotação ainda pode distorcer o resultado, qualquer ocorrência reprova
a validação final por padrão. A execução atual não registrou nenhuma.

## 11. Métricas de desempenho

Para cada estratégia são calculadas:

### Retorno total

```text
retorno_total = valor_final / capital_inicial - 1
```

### Retorno anualizado

```text
retorno_anual = (1 + retorno_total)^(1 / anos) - 1
```

### Volatilidade anualizada realizada

```text
vol_anual = desvio_padrão(retornos_diários) × raiz(365)
```

Essa é uma medida fora da amostra. Ela não precisa ser idêntica ao risco-alvo, que
é estimado no momento da otimização usando a janela histórica anterior.

### Máximo drawdown

É a maior queda percentual entre um pico do patrimônio e um vale posterior:

```text
drawdown(t) = (V(t) - máximo_histórico(t)) / máximo_histórico(t)
```

O capital inicial participa como primeiro pico, de modo que uma perda no primeiro
dia também seja contabilizada.

### Atividade de negociação

Também são registrados o turnover total, o turnover anualizado, o custo de
transação acumulado e o número de rebalanceamentos com negociação efetiva. Essas
métricas ganham importância com a frequência semanal, pois maior atualização dos
pesos pode elevar substancialmente os custos.

## 12. Testes estatísticos

Na análise principal, os retornos simples diários, incluindo custos, são compostos
em semanas não sobrepostas de segunda a domingo e convertidos em log-retornos.
Como robustez, o mesmo procedimento é repetido com meses civis. Para cada par de
estratégias, aplica-se o teste de Wilcoxon de postos sinalizados.

O Wilcoxon foi escolhido porque:

- compara observações pareadas do mesmo período;
- não exige normalidade dos retornos;
- é mais apropriado que um teste t quando existem assimetria, caudas pesadas e
  possíveis valores extremos.

São comparadas todas as combinações entre os três cenários e o benchmark. Como
várias hipóteses são testadas simultaneamente, também é calculado
o p-valor ajustado pelo método de Holm. Essa correção controla a probabilidade de
encontrar significância apenas por realizar muitos testes.

O relatório contém:

- frequência e número de períodos comparados;
- estatística de Wilcoxon;
- p-valor bruto;
- p-valor corrigido por Holm;
- diferença média e mediana;
- desvio-padrão, mínimo e máximo das diferenças;
- proporção de meses vencidos por cada estratégia.

## 13. Visualizações

O pipeline gera:

1. evolução do patrimônio de todas as estratégias;
2. drawdown ao longo do tempo;
3. representação simulada do espaço risco-retorno;
4. heatmap semanal de pesos para cada cenário;
5. heatmap do benchmark.

Os gráficos são salvos em `reports/figures` com resolução apropriada para uso no
artigo. O espaço risco-retorno usa somente a janela anterior ao último
rebalanceamento e é identificado como conjunto viável simulado, não como fronteira
eficiente exata.

## 14. Validações automáticas

O pipeline valida pesos, constituintes e diagnósticos antes do backtest. Depois do
backtest, valida os resultados completos antes de publicar qualquer artefato final.

### Pesos

- soma igual a 1;
- ausência de valores negativos;
- respeito aos limites configurados;
- peso zero em ativos fora da composição histórica vigente do NCI;
- valores finitos;
- existência de pelo menos um ativo selecionado.

### Otimização

- matriz PSD depois da correção;
- identificação de alvos de volatilidade inviáveis.
- verificação de que soluções factíveis respeitam o alvo de volatilidade.

### Backtest

- patrimônio sempre positivo;
- retornos dentro de intervalos plausíveis;
- igualdade entre retorno registrado e variação do patrimônio;
- drawdown nunca inferior a -100%;
- correspondência entre datas, pesos e retornos;
- reprovação de posições sem nova cotação, por padrão.

### Métricas

- volatilidade positiva;
- retorno acumulado dentro de uma faixa de sanidade.

Erros invalidam a execução. Alvos matematicamente inviáveis aparecem como avisos
explícitos; posições sem cotação são erros enquanto não houver uma regra metodológica
documentada para o evento.

## 15. Artefatos produzidos

### Dados processados

```text
data/processed/
```

### Pesos semanais

```text
results/weights/weights_<estratégia>.csv
```

### Diagnóstico da otimização

```text
results/optimization/optimization_diagnostics.csv
```

### Backtests e métricas

```text
results/metrics/backtest_<estratégia>.csv
results/metrics/metrics_summary.csv
```

### Testes estatísticos

```text
reports/tables/wilcoxon_weekly.csv
reports/tables/wilcoxon_weekly.md
reports/tables/wilcoxon_monthly_robustness.csv
reports/tables/wilcoxon_monthly_robustness.md
```

### Figuras

```text
reports/figures/
```

O gráfico `conjunto_viavel.png` combina uma amostra do conjunto viável com a
fronteira eficiente calculada pelo mesmo `EfficientFrontier` da otimização. Também
mostra as carteiras dos três cenários e o benchmark na última janela.

## 16. Interpretação da execução atual

A execução com estimadores padrão do PyPortfolioOpt, correção espectral da
covariância e maximização de retorno sob os limites de risco produziu:

| Estratégia | Retorno total | Retorno anualizado | Volatilidade realizada | Máximo drawdown |
|---|---:|---:|---:|---:|
| Risco 65% | 221,13% | 47,53% | 59,58% | -37,84% |
| Risco 85% | 113,00% | 28,67% | 73,27% | -43,23% |
| Risco 105% | 91,71% | 24,23% | 76,38% | -45,78% |
| Benchmark 1/N | 126,31% | 31,29% | 65,15% | -54,01% |

Os três cenários apresentaram volatilidade realizada crescente. Na otimização, o
limite ficou ativo em, respectivamente, 132, 91 e 4 das 157 janelas para os
alvos de 65%, 85% e 105%. Nas demais janelas, a carteira de máximo
retorno já estava abaixo do teto do cenário.

O cenário de 65% apresentou retorno acumulado superior ao benchmark e menor
volatilidade realizada. A significância desse resultado deve ser avaliada nas
tabelas semanal e mensal, sempre pelo p-valor corrigido por Holm; retorno acumulado
superior, isoladamente, não demonstra superioridade estatística consistente.

## 17. Limitações e decisões ainda sujeitas a estudo

### Retorno esperado

A média histórica é instável em criptoativos. Futuras análises podem comparar média
exponencial, shrinkage dos retornos ou Black-Litterman. Essas alternativas não devem
ser introduzidas apenas porque produzem melhor resultado na amostra atual.

### Concentração

A maximização pura de retorno gerou carteiras de ativo único em 25, 66 e 153 das
157 janelas nos cenários de 65%, 85% e 105%, respectivamente. Isso é uma
consequência possível da formulação escolhida e da instabilidade da média
histórica. Qualquer limite adicional de concentração deve
ser definido e justificado antes do experimento final, pois altera o conjunto de
carteiras factíveis.

### Definição dos cenários

Os alvos de 65%, 85% e 105% foram escolhidos com base na região factível
observada: o mínimo factível, um ponto intermediário e o extremo superior do
desenho. Eles devem ser congelados antes do experimento final.

### Ausência de caixa

Todo o capital é investido em criptoativos. Permitir caixa ou um ativo livre de
risco possibilitaria construir um perfil conservador com volatilidade muito menor.
Isso mudaria o universo do estudo e deve ser tratado como decisão metodológica.

### Universo de ativos

O pipeline usa a composição histórica trimestral publicada pela Nasdaq, com datas
efetivas point-in-time. A união dos constituintes serve apenas para coleta; pesos fora
da composição vigente são proibidos e validados automaticamente.

### MATIC

MATIC foi usado somente enquanto fazia parte do NCI, em Q2 e Q3 de 2024. Sua remoção
histórica em dezembro de 2024 elimina a dependência das cotações ausentes de 2025;
a execução atual não possui posições sem cotação.

### Significância estatística

O período fora da amostra contém 157 decisões de rebalanceamento, 156 retornos
semanais completos e 36 retornos mensais comparáveis. A última semana parcial é
excluída apenas do teste semanal. A análise semanal oferece maior resolução, mas retornos financeiros
podem apresentar dependência temporal; portanto, o aumento de observações não deve
ser interpretado como 156 replicações independentes. A análise mensal funciona como
verificação de robustez mais conservadora.

## 18. Reprodutibilidade

Para executar localmente:

```bash
make install
make run
```

Para ignorar o cache no intervalo solicitado:

```bash
python main.py --force-download
```

Para executar os testes:

```bash
make test
make lint
```

O `Makefile` centraliza os comandos de instalação, execução, lint e testes.

Para que uma execução seja considerada reproduzível, devem ser preservados:

1. o `config.yaml` utilizado;
2. a versão das dependências;
3. os arquivos de cache e suas fontes;
4. o diagnóstico das otimizações;
5. os pesos e resultados gerados;
6. o commit do código correspondente.

## 19. Princípio de interpretação

O objetivo científico não é ajustar o algoritmo até que ele necessariamente supere
o benchmark. O procedimento correto é definir a metodologia previamente, executar
o teste fora da amostra e relatar tanto resultados favoráveis quanto desfavoráveis.

Uma carteira superar o benchmark em retorno acumulado não prova superioridade. É
necessário observar risco, drawdown, custos, estabilidade dos pesos, diferentes
regimes de mercado e significância estatística. Essa distinção é essencial para que
o trabalho seja metodologicamente defensável.
