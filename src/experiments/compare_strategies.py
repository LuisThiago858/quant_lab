from __future__ import annotations
from src.data.datasets import load_features
from src.strategies.momentum import sinais_momentum_absoluto
from src.strategies.sma_cross import sma_crossover_signals
from src.strategies.zscore_reversion import sinais_reversao_zscore
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"

def avaliar(df_sinais, nome: str):
    resultado = run_backtest_long_only(
        df_sinais,
        initial_capital=10_000,
        fee_rate=0.001,
        slippage=0.0002,
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
    df_dados = load_features(SYMBOL, TF)
    estrategias = []
    # Momentum
    df_momentum = sinais_momentum_absoluto(df_dados, janela_lookback=72)
    estrategias.append(avaliar(df_momentum, "Momentum(72)"))
    # SMA
    df_sma = sma_crossover_signals(df_dados, short_window=20, long_window=100)
    estrategias.append(avaliar(df_sma, "SMA(20/100)"))
    # Mean Reversion
    df_mean_reversion = sinais_reversao_zscore(df_dados, janela_zscore=48, limiar_compra=-2.0, limiar_venda=2.0)
    estrategias.append(avaliar(df_mean_reversion, "ZScoreReversion(48)"))
    # Buy & Hold
    df_buy_hold = df_dados.copy()
    df_buy_hold["signal"] = 1
    df_buy_hold["position"] = 1
    estrategias.append(avaliar(df_buy_hold, "Buy&Hold"))
    print("\nComparação")
    for item in estrategias:
        print(
            f"{item['estrategia']:20s} - "
            f"Sharpe={item['sharpe']:.3f} | "
            f"CAGR={item['cagr']:.2%} | "
            f"MDD={item['mdd']:.2%}"
        )

if __name__ == "__main__":
    main()