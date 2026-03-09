from __future__ import annotations
from src.data.datasets import load_features
from src.strategies.zscore_reversion import sinais_reversao_zscore
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"

def main():
    df_dados = load_features(SYMBOL, TF)
    df_signal = sinais_reversao_zscore(
        df_dados,
        janela_zscore=48,
        limiar_compra=-2.0,
        limiar_venda=2.0,
    )
    resultado = run_backtest_long_only(
        df_signal,
        initial_capital=10_000,
        fee_rate=0.001,
        slippage=0.0002,
    )
    metricas = compute_metrics(resultado.equity, resultado.returns)
    print("Mean Reversion (Z-score)")
    print(f"Sharpe: {metricas.shape:.3f}")
    print(f"CAGR:   {metricas.cagr:.2%}")
    print(f"MDD:    {metricas.max_drawdown:.2%}")
    print(f"Equity final: {resultado.equity.iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()