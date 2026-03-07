# quant_lab 📈

Laboratório pessoal de **pesquisa em Trading Quantitativo** desenvolvido
em Python. O foco do projeto é construir uma base confiável de dados e
experimentação para testar hipóteses de mercado de forma reprodutível
--- tratando trading como um problema de **engenharia de dados +
estatística**, não apenas previsão de preço.

> ⚠️ Este projeto é educacional. Nada aqui constitui recomendação de
> investimento.

------------------------------------------------------------------------

## 🎯 Objetivos

-   Construir um pipeline confiável de dados financeiros (OHLCV)
-   Garantir qualidade e consistência dos dados de mercado
-   Produzir datasets prontos para backtesting
-   Implementar backtests reprodutíveis
-   Avaliar hipóteses de mercado com validação estatística
-   Evoluir para modelos quantitativos e ML

------------------------------------------------------------------------

## 🧱 Estrutura do Projeto

    quant_lab/
    ├─ data/
    │  ├─ raw/                       # dados brutos (parquet)
    │  └─ processed/                 # datasets com features (parquet)
    ├─ notebooks/
    │  ├─ backtest/                  # teste com taxas e sem taxas
    │  └─ strategies/                # estrategias
    ├─ src/
    │  ├─ data/
    │  │  ├─ binance_downloader.py   # download histórico via API da Binance
    │  │  ├─ datasets.py             # utilidades de leitura/escrita de datasets
    │  │  ├─ quality_checks.py       # verificação de gaps, duplicados e consistência
    │  │  └─ build_features.py       # engenharia de features financeiras
    │  └─ utils/
    │     └─ paths.py                # caminhos padronizados do projeto
    │
    │
    ├─ notebooks/                    # análises exploratórias (EDA)
    ├─ requirements.txt
    └─ .gitignore

------------------------------------------------------------------------

## 📊 Dados Utilizados

-   Ativo inicial: **BTCUSDT**
-   Timeframe: **1h**
-   Fonte: **Binance API**

Campos OHLCV: - Open (abertura) - High (máxima) - Low (mínima) - Close
(fechamento) - Volume

------------------------------------------------------------------------

## ⚙️ Pipeline de Dados

### 1) Download Histórico

Baixa candles OHLCV da Binance e salva em `data/raw/`.

### 2) Atualização Incremental

O sistema detecta automaticamente o último candle salvo e baixa **apenas
dados novos**, evitando reprocessar todo o histórico.

Benefícios: - Execução rápida - Dataset sempre atualizado - Custos
computacionais menores

### 3) Verificação de Qualidade

O módulo `quality_checks.py` valida: - gaps temporais - duplicatas -
inconsistências

Um relatório de qualidade é gerado em `data/processed/`.

### 4) Engenharia de Features

O módulo `build_features.py` cria métricas financeiras: - retorno
percentual (`ret`) - log-return (`log_ret`) - volatilidade rolling
(`vol_24`) - z-score de retornos (`zret_24`)

Saída principal:

    data/processed/BTCUSDT_1h_features.parquet

------------------------------------------------------------------------

## 🚀 Instalação

### 1) Clonar o repositório

``` bash
git clone https://github.com/LuisThiago858/quant_lab.git
cd quant_lab
```

### 2) Criar ambiente virtual

**Windows (PowerShell)**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1 
```

**Linux/Mac**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Instalar dependências

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Como Executar Semana 1

### Baixar parquet historico de BTCUSDT

``` bash
python -m src.data.binance_downloader
```

### Verifica a qualidade dos dados gerados, criando um relatorio atualizado, e uma visão geral da quantidade de missing data

``` bash
python -m src.data.quality_checks
```

### Gerar/atualizar dataset com features

``` bash
python -m src.data.build_features
```

### Carrega o arquivo de features em Parquet de um símbolo/timeframe e valida se o dataset está no formato certo (índice de tempo e colunas obrigatórias) para usar no backtest.

``` bash
python -m src.data.datasets
```

Isso irá: 1. Baixar dados faltantes da Binance 2. Validar qualidade 3.
Construir features 4. Salvar o dataset processado

## ▶️ Como Executar Semana 2

### Módulo de estratégia de SMA Cross (O cruzamento da Média Móvel Simples) que só gera sinais (não executa ordens de compra).

``` bash
python -m src.strategies.sma_cross
```

### Executar o motor de backtest vetorizado com pandas, já utilizando retorno e posições gerados no passo anterior, e utilizando um capital inicial e retornando o capital final.

``` bash
python -m src.backtest.engine
```

### Executar o módulo metricas que recebe uma equity curve e/ou retornos e devolve Sharpe, Max Drawdown e CAGR.

``` bash
python -m src.backtest.metrics
```

### Gerar um mini report com gráfico entre as estratégias com custo e sem custo do SMA e Buy & Hold

``` bash
python -m src.backtest.report
```

### Testar estrategias de SMA como periodos de curto e longo prazo diferentes
``` bash
python -m src.experiments.sma_grid_search
```

### Executar código testa várias configurações da estratégia em dados passados (treino), escolhe a melhor e depois verifica como ela se comporta em dados futuros (teste). Isso ajuda a ver se a estratégia realmente funciona fora da amostra e reduz o risco de overfitting.

``` bash
python -m src.experiments.walkforward_sma
```
## ▶️ Como Executar Semana 3

### Executar simulação de Monte Carlo / Bootstrap para avaliar a distribuição de Sharpe, CAGR e Drawdown da estratégia a partir de reamostragens dos retornos históricos.

``` bash
python -m src.analysis.monte_carlo_bootstrap
```

### Gerar um relatório textual de robustez com intervalos de confiança (p05, p50 e p95) a partir dos resultados do bootstrap.

``` bash
python -m src.analysis.bootstrap_report
```

### Avaliar a estabilidade dos parâmetros da estratégia SMA em uma grade de combinações próximas e gerar um heatmap de Sharpe para identificar regiões mais robustas.

``` bash
python -m src.experiments.parameter_stability
```

### Executar validação rolling walk-forward com múltiplas janelas móveis, recalibrando a melhor configuração no treino e validando no teste subsequente.

``` bash
python -m src.experiments.rolling_walkforward
```

### Estimar a Probability of Backtest Overfitting (PBO) de forma prática, verificando com que frequência a melhor estratégia do treino cai para posições ruins no ranking do teste.

``` bash
python -m src.analysis.pbo_estimation
```

### Comparar a estratégia SMA com Buy & Hold e com estratégias aleatórias calibradas por turnover, para avaliar se a estratégia supera o acaso de forma consistente.

``` bash
python -m src.analysis.random_baseline
```

------------------------------------------------------------------------

## 📥 Carregar Dataset no Código

``` python
from src.data.datasets import load_features

df = load_features("BTCUSDT", "1h")
print(df.tail())
```

------------------------------------------------------------------------

## 🧪 Roadmap

-   Motor de backtesting
-   Métricas de performance (Sharpe, drawdown, win rate)
-   Walk-forward validation
-   Otimização de parâmetros
-   Múltiplos ativos
-   Integração com Machine Learning

------------------------------------------------------------------------

## 👨‍💻 Autor

Projeto desenvolvido como estudo de engenharia de dados aplicada a
finanças quantitativas.
