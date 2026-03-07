from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF="1h"

CAPITAL_INICIAL = 10_000
TAXA_CORRETAGEM = 0.001
DESVIO_EXECUCAO = 0.0002

# candidatdos a parametros baseado na analise de estabilidades
COMBINACOES_PARAMETROS = [
    (20, 80), (22, 80), (24, 80),
    (18, 100), (20, 100),
]

#Proporcoes da janela movel a rolling window
PROPORCAO_TREINO = 0.50 #50% dos dados para otimizaçao
PROPORCAO_TESTE = 0.10 #10% para validação fora da amostra o Out-Of-Sample
PROPORCAO_PASSO = 0.10 #O quanto a janela "anda" para a frente a cada rodada

def calcular_performance_configuracao(df_dados: pd.DataFrame, curta: int, longa: int) -> dict:
    """"Roda o backtest para uma combinação e retorna métricas resumidas."""
    df_com_sinais = sma_crossover_signals(df_dados, curta, longa)

    resultado = run_backtest_long_only(
        df_com_sinais,
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
        "drawdown_maximo": metricas.max_drawdown,
    }

def selecionar_melhor_parametro_no_treino(df_treino: pd.DataFrame)->dict:
    """Teste todas as combinaçoes no treino e retorna a vencedora pelo Sharpe."""

    resultados_treino = []
    for curta, longa in COMBINACOES_PARAMETROS:
        performance = calcular_performance_configuracao(df_treino, curta, longa)
        resultados_treino.append(performance)
    #Ordena pelo Sharpe e pega o melhor
    df_ranking = pd.DataFrame(resultados_treino).sort_values("sharpe", ascending=False)
    return df_ranking.iloc[0].to_dict()

def main():
    #Carrega e garante ordem cronologica
    df_completo = load_features(SYMBOL, TF).sort_index()
    total_pontos = len(df_completo)
    tamanho_treino = int(total_pontos * PROPORCAO_TREINO)
    tamanho_teste = int(total_pontos * PROPORCAO_TESTE)
    tamanho_passo = int(total_pontos * PROPORCAO_PASSO)

    historico_walkfoward = []
    rodada=1
    ponteiro_inicio=0

    while True:
        ponteiro_fim_treino = ponteiro_inicio + tamanho_treino
        ponteiro_inicio_teste = ponteiro_fim_treino
        ponteiro_fim_teste = ponteiro_inicio_teste + tamanho_teste

        if ponteiro_fim_teste > total_pontos:
            break

        df_janela_treino = df_completo.iloc[ponteiro_inicio:ponteiro_fim_treino].copy()
        df_janela_teste = df_completo.iloc[ponteiro_inicio_teste: ponteiro_fim_teste].copy()

        print(f"\n Rodada {rodada}")
        #otimização para encontrar o melhor parametro no passado recente
        melhor_config = selecionar_melhor_parametro_no_treino(df_janela_treino)
        curta_vencedora = int(melhor_config["media_curta"])
        longa_vencedora = int(melhor_config["media_longa"])
        #validacao aplica no futuro imediato (In-Sample -> Out-of-Sample)
        performance_teste = calcular_performance_configuracao(df_janela_teste, curta_vencedora, longa_vencedora)
        #baseline de como seria apenas segurar o ativo buy e hold nesse periodo
        retornos_ativos = df_janela_teste["ret"].fillna(0.0)
        patrimonio_bh = CAPITAL_INICIAL * (1.0 + retornos_ativos).cumprod()
        metricas_bh = compute_metrics(patrimonio_bh, retornos_ativos)

        historico_walkfoward.append({
            "rodada": rodada,
            "periodo_teste": f"{df_janela_teste.index.min().date()} | {df_janela_teste.index.max().date()}",
            "config_vencedora": f"SMA({curta_vencedora}/{longa_vencedora})",
            "sharpe_treino": melhor_config["sharpe"],
            "sharpe_teste": performance_teste["sharpe"],
            "cagr_teste": performance_teste["cagr"],
            "bh_sharpe": metricas_bh.shape,
        })

        rodada=rodada+1
        ponteiro_inicio = ponteiro_inicio + tamanho_passo #move a janela para frente
    #consolidação dos resultados
    df_final = pd.DataFrame(historico_walkfoward)
    print("\n"+"="*50)
    print("Resumo final Walk-Forward")
    print("="*50)
    print(df_final)

    #exportacao
    caminho_saida = Path("data/processed/walkfoward_rolling_summary.csv")
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(caminho_saida, index=False)
    print(f"\n Relatorio salvo em: {caminho_saida}")

if __name__ == "__main__":
    main()

    