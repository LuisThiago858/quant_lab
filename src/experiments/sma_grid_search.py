from __future__ import annotations
import pandas as pd

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

#configurações a serem testadas com o sma
CONFIGS = [
    (10, 50),
    (20, 100),
    (50, 200),
    (30, 150),
    (5, 200),
]

def run_experiment(symbol="BTCUSDT", timeframe="1h"):
    df_experiment = load_features(symbol, timeframe)

    result=[]
    for short, long in CONFIGS:
        print(f"Rodando SMA({short}/{long})...")

        df_signal = sma_crossover_signals(df_experiment, short, long)

        res = run_backtest_long_only(
            df_signal,
            initial_capital=10_000,
            fee_rate=0.001,
            slippage=0.0002,
        )

        metrics = compute_metrics(res.equity, res.returns)

        result.append({
            "short": short,
            "long": long,
            "sharpe": metrics.shape,
            "cagr": metrics.cagr,
            "mdd": metrics.max_drawdown,
        })

    df_resultados = pd.DataFrame(result).sort_values("sharpe", ascending=False)
    return df_resultados

if __name__ == "__main__":
    table = run_experiment()
    print("\n RESULTADO")
    print(table)

    table.to_csv("data/processed/sma_experimentos_resultados.csv", index=False)
    print("\n Resultados salvos em data/processed/sma_experimentos_resultados.csv")