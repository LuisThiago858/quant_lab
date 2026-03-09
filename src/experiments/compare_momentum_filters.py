from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data.datasets import load_features
from src.strategies.momentum import (
    sinais_momentum_absoluto,
    momentum_com_filtro_sma,
    momentum_com_filtro_volatilidade,
    momentum_com_limiar_de_forca,
)
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"
CAPITAL_INICIAL = 10_000
TAXA_CORRETAGEM = 0.001
DESVIO_EXECUCAO = 0.0002

def avaliar_estrategia(df_sinais: pd.DataFrame, nome: str) -> dict:
    """
    Executa backtest e calcula métricas.
    """
    resultado = run_backtest_long_only(
        df_sinais,
        initial_capital=CAPITAL_INICIAL,
        fee_rate=TAXA_CORRETAGEM,
        slippage=DESVIO_EXECUCAO,
    )

    metricas = compute_metrics(resultado.equity, resultado.returns)

    return {
        "estrategia": nome,
        "sharpe": metricas.shape,
        "cagr": metricas.cagr,
        "mdd": metricas.max_drawdown,
        "equity_final": float(resultado.equity.iloc[-1]),
    }


def main():
    df = load_features(SYMBOL, TF).sort_index()

    estrategias = {
        "Momentum_48": sinais_momentum_absoluto(df, janela_lookback=48),
        "Momentum_72": sinais_momentum_absoluto(df, janela_lookback=72),

        "Momentum_48_SMA100": momentum_com_filtro_sma(
            df,
            janela_lookback=48,
            janela_sma=100,
        ),
        "Momentum_72_SMA100": momentum_com_filtro_sma(
            df,
            janela_lookback=72,
            janela_sma=100,
        ),
        "Momentum_48_Limiar1pct": momentum_com_limiar_de_forca(
            df,
            janela_lookback=48,
            limiar_minimo=0.01,
        ),
        "Momentum_72_Limiar1pct": momentum_com_limiar_de_forca(
            df,
            janela_lookback=72,
            limiar_minimo=0.01,
        ),
        "Momentum_48_VolLow": momentum_com_filtro_volatilidade(
            df,
            janela_momentum=48,
            percentil_corte=0.50,
            modo_filtro="low",
        ),
        "Momentum_48_VolHigh": momentum_com_filtro_volatilidade(
            df,
            janela_momentum=48,
            percentil_corte=0.50,
            modo_filtro="high",
        ),
        "Momentum_72_VolLow": momentum_com_filtro_volatilidade(
            df,
            janela_momentum=72,
            percentil_corte=0.50,
            modo_filtro="low",
        ),
        "Momentum_72_VolHigh": momentum_com_filtro_volatilidade(
            df,
            janela_momentum=72,
            percentil_corte=0.50,
            modo_filtro="high",
        ),
    }

    resultados = []
    print("\nIniciando comparação entre variantes de momentum...\n")
    for nome, df_sinais in estrategias.items():
        print(f"Testando: {nome}")
        resultado = avaliar_estrategia(df_sinais, nome)
        resultados.append(resultado)
    df_resultados = pd.DataFrame(resultados).sort_values("sharpe", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("Comparação das variantes de Momentum")
    print("=" * 70)
    print(df_resultados.to_string(index=False))
    #exportar
    diretorio_saida = Path("data/processed")
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    arquivo = diretorio_saida / "momentum_filters_comparison.csv"
    df_resultados.to_csv(arquivo, index=False)
    print(f"\nResultado salvo em: {arquivo}")

if __name__ == "__main__":
    main()