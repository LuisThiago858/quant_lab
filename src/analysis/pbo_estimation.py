from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"

# Configurações Financeiras
CAPITAL_INICIAL = 10_000
TAXA_CORRETAGEM = 0.001
DESVIO_EXECUCAO = 0.0002

#estrategias candidatas
UNIVERSO_CONFIGURACOES = [
    (16, 80), (18, 80), (20, 80), (22, 80), (24, 80),
    (18, 100), (20, 100), (22, 100), (16, 110), (20, 110)
]

#divisao dos dados em fatias temporais
NUMERO_BLOCOS=8

def avaliar_desempenho_config(df_dados: pd.DataFrame, curta: int, longa: int)->dict:
    """Roda backtest e extrai metricas para uma unica combinacao."""

    df_sinais = sma_crossover_signals(df_dados, curta, longa)
    resultado = run_backtest_long_only(
        df_sinais,
        initial_capital=CAPITAL_INICIAL,
        fee_rate=TAXA_CORRETAGEM,
        slippage=DESVIO_EXECUCAO,
    )

    metricas = compute_metrics(resultado.equity, resultado.returns)
    return {
        "media_curta": curta,
        "media_longa": longa,
        "sharpe": metricas.shape,
        "cagr": metricas.cagr,
        "mdd": metricas.max_drawdown,
    }

def dividir_dados_em_blocos(df_completo: pd.DataFrame, n_blocos: int)->list[pd.DataFrame]:
    """Fatia o dataframe original em N partes iguais cronologicas"""

    tamanho_total = len (df_completo)
    tamanho_bloco = tamanho_total // n_blocos
    lista_blocos =[]

    for i in range(n_blocos):
        inicio = i*tamanho_bloco
        #Garante que o ultimo bloco pegue o resto dos dados
        fim = (i+1) * tamanho_bloco if i<n_blocos-1 else tamanho_total
        lista_blocos.append(df_completo.iloc[inicio:fim].copy())

    return lista_blocos

def avaliar_universo_inteiro(df_segmento: pd.DataFrame)-> pd.DataFrame:
    """"Calcula o Sharpe de todas as configuracoes canditadas em um segmento de dados."""
    resultados = []
    for curta, longa in UNIVERSO_CONFIGURACOES:
        performance = avaliar_desempenho_config(df_segmento, curta, longa)
        resultados.append(performance)
    return pd.DataFrame(resultados)

def main():
    df_historico = load_features(SYMBOL, TF).sort_index()
    lista_blocos = dividir_dados_em_blocos(df_historico, NUMERO_BLOCOS)

    historico_pbo = [] #probabilidade do backteste sobreajustar
    contador_overfit = 0
    total_rodadas = 0

    #logica de janela para combinar blocos para treino e blocos subsequentes para teste
    for i in range(0, NUMERO_BLOCOS - 5):
        df_treino = pd.concat(lista_blocos[i:i+4]).sort_index()
        df_teste = pd.concat(lista_blocos[i+4:i+6]).sort_index()

        print(f"\nAnalise PBO: rodadas {total_rodadas+1}")

        #gera o ranking de todas as estrategias tanto no treino quando no teste
        ranking_treino = avaliar_universo_inteiro(df_treino).sort_values("sharpe", ascending=False).reset_index(drop=True)
        ranking_teste = avaliar_universo_inteiro(df_teste).sort_values("sharpe", ascending=False).reset_index(drop=True)
        #Identifica a campea do treino 
        melhor_treino = ranking_treino.iloc[0]
        c_vencedora, l_vencedora = int(melhor_treino["media_curta"]), int(melhor_treino["media_longa"])
        #verifica onde essa campea ficou no ranking do teste o out-of-sample
        filtro_vencedora=(ranking_teste["media_curta"]==c_vencedora) & (ranking_teste["media_longa"]==l_vencedora)
        posicao_no_teste = int(ranking_teste[filtro_vencedora].index[0]) + 1
        percentil_teste = posicao_no_teste/len(ranking_teste)
        #O criterio de overfitting e se a melhor do treino cair abaxio da mediana no teste
        falhou_no_teste = percentil_teste >0.5
        if falhou_no_teste:
            contador_overfit = contador_overfit + 1
        total_rodadas = total_rodadas + 1

        print(f"Melhor no Treino: SMA({c_vencedora}/{l_vencedora})")
        print(f"Posicao no Teste: {posicao_no_teste}/{len(ranking_teste)} (Percentil: {percentil_teste:.2f})")

        historico_pbo.append({
            "rodada": total_rodadas,
            "config": f"{c_vencedora}/{l_vencedora}",
            "sharpe_treino": melhor_treino["sharpe"],
            "percentil_teste": percentil_teste,
            "houve_overfit": falhou_no_teste,
        })

    #resultado final ja calculado
    probabilidade_overfit = contador_overfit/total_rodadas
    df_final = pd.DataFrame(historico_pbo)

    print("\n" + "="*50)
    print(f"Probabilidade de overfit PBO: {probabilidade_overfit:.2%}")
    print("="*50)

    #exportar
    caminho_csv = Path("data/processed/pbo_avaliacao.csv")
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(caminho_csv, index=False)
    print(f"Resultado salvo em {caminho_csv}")

if __name__ == "__main__":
    main()