from src.data.datasets import load_features
from src.strategies.momentum import sinais_momentum_absoluto
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF="1h"

df_dados = load_features(SYMBOL, TF)
df_momentum = sinais_momentum_absoluto(df_dados, janela_lookback=24)

resultado_momentum = run_backtest_long_only(
    df_momentum,
    initial_capital=10_000,
    fee_rate=0.001,
    slippage=0.0002,
)

metricas_momentum = compute_metrics(resultado_momentum.equity, resultado_momentum.returns)

print("Momentum(24)")
print(f"Sharpe: {metricas_momentum.shape:.3f}")
print(f"CAGR:   {metricas_momentum.cagr:.2%}")
print(f"MDD:    {metricas_momentum.max_drawdown:.2%}")
print(f"Equity final: {resultado_momentum.equity.iloc[-1]:,.2f}")
