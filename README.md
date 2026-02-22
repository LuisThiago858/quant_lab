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
    ├─ src/
    │  ├─ data/
    │  │  ├─ binance_downloader.py   # download histórico via API da Binance
    │  │  ├─ datasets.py             # utilidades de leitura/escrita de datasets
    │  │  ├─ quality_checks.py       # verificação de gaps, duplicados e consistência
    │  │  └─ build_features.py       # engenharia de features financeiras
    │  └─ utils/
    │     └─ paths.py                # caminhos padronizados do projeto
    │
    ├─ data/
    │  ├─ raw/                       # dados brutos (parquet)
    │  └─ processed/                 # datasets com features (parquet)
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
.\\.venv\\Scripts\\activate
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

## ▶️ Como Executar

### Baixar parquet historico de BTCUSDT
``` bash
python -m src.data.binance_downloader
```

### Gerar/atualizar dataset com features

``` bash
python -m src.data.build_features
```

### Verifica a qualidade dos dados gerados, criando um relatorio atualizado, e uma visão geral da quantidade de missing data

``` bash
python -m src.data.quality_checks
```

### Carrega o arquivo de features em Parquet de um símbolo/timeframe e valida se o dataset está no formato certo (índice de tempo e colunas obrigatórias) para usar no backtest.

``` bash
python -m src.data.datasets
```

Isso irá: 1. Baixar dados faltantes da Binance 2. Validar qualidade 3.
Construir features 4. Salvar o dataset processado

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
