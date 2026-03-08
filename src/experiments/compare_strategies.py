from src.data.datasets import load_features
from src.strategies.momentum import sinais_momentum_absoluto
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL="BTCUSDT"
TF="1h"

df_dados = load_features(SYMBOL, TF)

#momentum
df_momentum = sinais_momentum_absoluto(df_dados, janela_lookback=168)
resultado_momentum = run_backtest_long_only(df_momentum, initial_capital=10_000, fee_rate=0.001, slippage=0.0002)
metricas_momentum = compute_metrics(resultado_momentum.equity, resultado_momentum.returns)

# SMA
df_sma = sma_crossover_signals(df_dados, short_window=20, long_window=100)
resultado_sma = run_backtest_long_only(df_sma, initial_capital=10_000, fee_rate=0.001, slippage=0.0002)
metricas_sma = compute_metrics(resultado_sma.equity, resultado_sma.returns)

# Buy & Hold
df_bh = df_dados.copy()
df_bh["signal"] = 1
df_bh["position"] = 1
resultado_bh = run_backtest_long_only(df_bh, initial_capital=10_000, fee_rate=0.001, slippage=0.0002)
metricas_bh = compute_metrics(resultado_bh.equity, resultado_bh.returns)

print("Comparação")
print(f"Momentum(24) - Sharpe={metricas_momentum.shape:.3f} | CAGR={metricas_momentum.cagr:.2%} | MDD={metricas_momentum.max_drawdown:.2%}")
print(f"SMA(20/100)  - Sharpe={metricas_sma.shape:.3f} | CAGR={metricas_sma.cagr:.2%} | MDD={metricas_sma.max_drawdown:.2%}")
print(f"Buy&Hold     - Sharpe={metricas_bh.shape:.3f} | CAGR={metricas_bh.cagr:.2%} | MDD={metricas_bh.max_drawdown:.2%}")