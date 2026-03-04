#pegar os retonros da estrategia anterior e simular vários nmercados alternativis
#criar simulações da equity curve, pegado o strategy_returns, reamostrar com reposição, reconstruir a equity e repetir 500 vezes

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL="BTCUSDT"
TF="1h"
SHORT_W=20
LONG_W=100

CAPITAL_INICIAL=10_000
FEE=0.001
SLIP=0.0002

NUMERO_SIMULACOES=500
SEED=42
PERIODOS_POR_ANO=8760 #1h

def equity_from_returns(series_returnos: pd.Series, capital_initial: float) -> pd.Series:
    """
    Reconstruir equity curve a partir de uma série de retornos percentuais.
    """
    retornos = series_returnos.fillna(0.0).astype(float).to_numpy()
    equity_curve = capital_initial * np.cumprod(1.0 + retornos)

    return pd.Series(equity_curve)

def bootstrap_simulation_metrics(
    returns: pd.Series,
    num_simulacoes: int=500,
    capital_inicial: float=10_000,
    periodos_por_ano: int=8760,
    seed: int = 42,
)->pd.DataFrame:
    """"
    Bootstrap com reposição:
        -amostra retornos com reposição (mesmo tamanho)
        -Reconstroi equity
        -calcula Sharpe e CAGR
    """

    retornos_limpos = returns.dropna().astype(float).to_numpy()
    tamanho_amostra = len(retornos_limpos)

    if tamanho_amostra<10:
        raise ValueError("Retornos insuficientes para bootstrap(precisa de uma série maior).")
    
    gerador_aleatorio = np.random.default_rng(seed)
    resultados_simulados= []

    for index in range(num_simulacoes):
        #sorteia novos indices aleatorios (bootstrap com reposição)
        indices_aleatorios = gerador_aleatorio.integers(0, tamanho_amostra, size=tamanho_amostra)
        retornos_simulados = pd.Series(retornos_limpos[indices_aleatorios])
        equity_simulada = equity_from_returns(retornos_simulados, capital_inicial)

        #calcula métricas para esta simulação específica
        metricas = compute_metrics(equity_simulada, retornos_simulados, periodos_por_ano=periodos_por_ano)
        resultados_simulados.append({
            "simulacao_id": index,
            "sharpe": metricas.shape,
            "cagr": metricas.cagr,
            "mdd": metricas.max_drawdown,
            "patrimonio_final": float(equity_simulada.iloc[-1]),
        })
    return pd.DataFrame(resultados_simulados)
    
def resumir_distribuicao_estatistica(df_simulations: pd.DataFrame, coluna: str) -> dict:
    """
    Analisa a série de resultados simulados e extrai os principais 
    pontos estatísticos (média e decis de confiança).
    """
    serie_dados = df_simulations[coluna].dropna()

    return {
        "media": float(serie_dados.mean()),
        "pior_cenario_5pct": float(serie_dados.quantile(0.05)),
        "mediana_50pct": float(serie_dados.quantile(0.50)),
        "melhor_cenario_95pct": float(serie_dados.quantile(0.95)),
    }

def main():
    dados_historicos = load_features(SYMBOL, TF)
    dados_com_sinais = sma_crossover_signals(dados_historicos, SHORT_W, LONG_W)

    data_inicio = dados_com_sinais.index.min()
    data_fim = dados_com_sinais.index.max()
    ponto_de_corte = data_inicio + (data_fim - data_inicio) * 0.70

    dados_treino = dados_com_sinais.loc[:ponto_de_corte].copy()
    #periodo de treino já com as taxas
    resultado_backtest = run_backtest_long_only(
        dados_treino,
        initial_capital=CAPITAL_INICIAL,
        fee_rate=FEE,
        slippage=SLIP,
    )
    retornos_estrategia = resultado_backtest.returns.dropna()
    metricas_reais = compute_metrics(
        resultado_backtest.equity, 
        resultado_backtest.returns, 
        periodos_por_ano=PERIODOS_POR_ANO
    )

    print(f"Período de Treino: {dados_treino.index.min()} -> {dados_treino.index.max()} (n={len(dados_treino)})")
    print(f"Total de retornos gerados: {len(retornos_estrategia)} pontos\n")

    print("Resultado Real Observado (TREINO)")
    print(f"SMA({SHORT_W}/{LONG_W}) Sharpe={metricas_reais.shape:.3f} "
          f"CAGR={metricas_reais.cagr:.2%} MDD={metricas_reais.max_drawdown:.2%}\n")

    # inicia as Simulações de Monte Carlo via Bootstrap
    df_simulacoes = bootstrap_simulation_metrics(
        returns=retornos_estrategia,
        num_simulacoes=NUMERO_SIMULACOES,
        capital_inicial=CAPITAL_INICIAL,
        periodos_por_ano=PERIODOS_POR_ANO,
        seed=SEED,
    )

    # resumo estatístico dos 500 "futuros alternativos" (Intervalos de Confiança)
    estatisticas_sharpe = resumir_distribuicao_estatistica(df_simulacoes, "sharpe")
    estatisticas_cagr = resumir_distribuicao_estatistica(df_simulacoes, "cagr")
    estatisticas_mdd = resumir_distribuicao_estatistica(df_simulacoes, "mdd")

    print("Bootstrap (Distribuição Simulada)")
    print(f"Sharpe: média={estatisticas_sharpe['media']:.3f} | p05={estatisticas_sharpe['pior_cenario_5pct']:.3f} | p95={estatisticas_sharpe['melhor_cenario_95pct']:.3f}")
    print(f"CAGR:   média={estatisticas_cagr['media']:.2%} | p05={estatisticas_cagr['pior_cenario_5pct']:.2%} | p95={estatisticas_cagr['melhor_cenario_95pct']:.2%}")
    print(f"MDD:    média={estatisticas_mdd['media']:.2%} | p05={estatisticas_mdd['pior_cenario_5pct']:.2%} | p95={estatisticas_mdd['melhor_cenario_95pct']:.2%}\n")

    # ranking: Onde o resultado real se encaixa na curva de probabilidade?
    # isso ajuda a detectar se o resultado real foi "fora da curva" por sorte.
    percentil_sharpe_real = float((df_simulacoes["sharpe"] <= metricas_reais.shape).mean())
    percentil_cagr_real = float((df_simulacoes["cagr"] <= metricas_reais.cagr).mean())

    print("Posição do resultado observado dentro das simulações")
    print(f"O Sharpe real está acima de {percentil_sharpe_real:.1%} das simulações.")
    print(f"O CAGR real está acima de {percentil_cagr_real:.1%} das simulações.\n")

    # salvar resultados para auditoria posterior
    diretorio_saida = Path("data/processed")
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    caminho_csv = diretorio_saida / f"bootstrap_sma_{SYMBOL}_{TF}.csv"
    df_simulacoes.to_csv(caminho_csv, index=False)
    print(f"✅ Resultados salvos em: {caminho_csv}")

    # visualização Gráfica
    # o histograma mostra a frequência dos resultados e a linha tracejada é o seu resultado real.
    
    plt.figure(figsize=(12, 5))
    
    # subplot para Sharpe
    plt.subplot(1, 2, 1)
    df_simulacoes["sharpe"].hist(bins=40, alpha=0.7, color='steelblue', edgecolor='white')
    plt.axvline(metricas_reais.shape, color='red', linestyle="--", linewidth=2, label="Real")
    plt.title(f"Bootstrap Sharpe - {SYMBOL}")
    plt.xlabel("Índice Sharpe")
    plt.legend()

    # subplot para CAGR
    plt.subplot(1, 2, 2)
    df_simulacoes["cagr"].hist(bins=40, alpha=0.7, color='seagreen', edgecolor='white')
    plt.axvline(metricas_reais.cagr, color='red', linestyle="--", linewidth=2, label="Real")
    plt.title(f"Bootstrap CAGR - {SYMBOL}")
    plt.xlabel("Retorno Anualizado (CAGR)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
