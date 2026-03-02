from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

def drawdown_series(equity: pd.Series) -> pd.Series:
    pico = equity.cummax()
    return equity / pico - 1.0

def main():
    symbol = "BTCUSDT"
    tf = "1h"
    short_w, long_w = 20, 100
    initial = 10_000

    df_atualizado = load_features(symbol, tf)
    df_signal = sma_crossover_signals(df_atualizado, short_w, long_w)

    #Estrategia sem custo e com cuwsto
    res_nocost = run_backtest_long_only(df_signal, initial_capital = initial, fee_rate=0.0, slippage=0.0)
    res_cost = run_backtest_long_only(df_signal, initial_capital=initial, fee_rate=0.001, slippage=0.0002)

    met_nocost = compute_metrics(res_nocost.equity, res_nocost.returns)
    met_cost = compute_metrics(res_cost.equity, res_cost.returns)

    #Buy & Hold
    bh_returns = df_signal["ret"].fillna(0.0)
    bh_equity = initial * (1.0 + bh_returns).cumprod()
    met_bh = compute_metrics(bh_equity, bh_returns)

    #resumo
    lines = []
    lines.append(f"Relatório: SMA({short_w}/{long_w}) {symbol} {tf}")
    lines.append(f"Período: {df_signal.index.min()} -> {df_signal.index.max()}")
    lines.append("")

    lines.append("=== Estratégia SEM custo ===")
    lines.append(f"Sharpe: {met_nocost.shape:.3f}")
    lines.append(f"CAGR:   {met_nocost.cagr:.2%}")
    lines.append(f"MDD:    {met_nocost.max_drawdown:.2%} ({met_nocost.max_drawdown_start} -> {met_nocost.max_drawdown_end})")
    lines.append("")

    lines.append("=== Estratégia COM custo ===")
    lines.append(f"Sharpe: {met_cost.shape:.3f}")
    lines.append(f"CAGR:   {met_cost.cagr:.2%}")
    lines.append(f"MDD:    {met_cost.max_drawdown:.2%} ({met_cost.max_drawdown_start} -> {met_cost.max_drawdown_end})")
    lines.append("")

    lines.append("=== Buy & Hold ===")
    lines.append(f"Sharpe: {met_bh.shape:.3f}")
    lines.append(f"CAGR:   {met_bh.cagr:.2%}")
    lines.append(f"MDD:    {met_bh.max_drawdown:.2%} ({met_bh.max_drawdown_start} -> {met_bh.max_drawdown_end})")
    lines.append("")

    lines.append("Resumo interpretativo:")
    lines.append("- Sem custos, a estratégia parece muito melhor do que realmente é.")
    lines.append("- Com custos, o turnover reduz Sharpe e CAGR; Buy&Hold vence em retorno/Sharpe, mas perde em drawdown.")

    report_text = "\n".join(lines)
    print(report_text)

    out_path = Path("data/processed") / f"report_SMA{short_w}_{long_w}_{symbol}_{tf}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nRelatório salvo em: {out_path}")

    # Plotando o gráfico Equity e Drawdown
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    res_nocost.equity.plot(ax=ax1)
    res_cost.equity.plot(ax=ax1)
    bh_equity.plot(ax=ax1)
    ax1.set_title(f"Equity: SMA({short_w}/{long_w}) vs Buy & Hold ({symbol} {tf})")
    ax1.legend(["SMA sem custo", "SMA com custo", "Buy & Hold"])
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    drawdown_series(res_nocost.equity).plot(ax=ax2)
    drawdown_series(res_cost.equity).plot(ax=ax2)
    drawdown_series(bh_equity).plot(ax=ax2)
    ax2.set_title(f"Drawdown: SMA({short_w}/{long_w}) vs Buy & Hold ({symbol} {tf})")
    ax2.legend(["SMA sem custo", "SMA com custo", "Buy & Hold"])
    plt.show()


if __name__ == "__main__":
    main()