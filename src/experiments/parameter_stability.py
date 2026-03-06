from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"

# Grades de Parâmetros para Teste de Estresse
VALORES_MEDIA_CURTA = [16, 18, 20, 22, 24]
VALORES_MEDIA_LONGA = [80, 90, 100, 110, 120]

CAPITAL_INICIAL = 10_000
TAXA_CORRETAGEM = 0.001
DESVIO_EXECUCAO = 0.0002

def avaliar_configuracao(dados_precos, janela_curta, janela_longa):
    """Executar backtest e retorna o indice para uma combinação especifica"""

    dados_sinais = sma_crossover_signals(dados_precos, janela_curta, janela_longa)

    resultado = run_backtest_long_only(
        dados_sinais,
        initial_capital=CAPITAL_INICIAL,
        fee_rate=TAXA_CORRETAGEM,
        slippage=DESVIO_EXECUCAO,
    )

    metricas = compute_metrics(resultado.equity, resultado.returns)

    return metricas.shape

def main():
    dados_historicos = load_features(SYMBOL, TF)
    lista_resultados = []
    for curta in VALORES_MEDIA_CURTA:
        for longa in VALORES_MEDIA_LONGA:
            if curta >= longa:
                continue

            print(f"Testando combinação: Curta {curta} e Longa {longa}...")

            indice_sharpe = avaliar_configuracao(dados_historicos, curta, longa)

            lista_resultados.append({
                "janela_curta": curta,
                "janela_longa": longa,
                "sharpe": indice_sharpe,
            })

    df_estabilidade = pd.DataFrame(lista_resultados)

    print("\nMatriz de resultados:")
    print(df_estabilidade)

    #visualização em um mapa de calor em uma matriz
    matriz_pivot = df_estabilidade.pivot(index="janela_curta", columns="janela_longa", values="sharpe")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz_pivot,
        annot=True,
        cmap="RdYlGn",
        center=0,
        fmt=".2f",
    )

    plt.title(f"Estabilidade de parâmetros - Sharpe \n Ativo: {SYMBOL} | TF{TF}:")
    plt.xlabel("Janela Média Longa")
    plt.ylabel("Janela Média Curta")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()