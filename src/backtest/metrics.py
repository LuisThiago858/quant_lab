from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)

class Metrics:
    shape: float
    cagr: float
    max_drawdown: float
    max_drawdown_start: pd.Timestamp | None
    max_drawdown_end: pd.Timestamp |None


def shape_ratio(returns: pd.Series, periodos_por_ano: int = 8760) -> float:
    """
    Shape (fr=0): mean(ret)/std(ret) * sqrt(periodos_por_ano)
    """

    returns = returns.dropna().astype(float)

    if len(returns)<2:
        return float("nan")
    
    #para evitar divisão por zero
    std = returns.std(ddof=1)

    if std == 0 or np.isnan(std):
        return float("nan")
    
    return (returns.mean()/std) * np.sqrt(periodos_por_ano)

def max_drawdon(equity: pd.Series)-> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """"
    Max Drawdown a partir da equity curve:
        Drawdown = equity / running_max - 1
    Retorna: (mdd, start, end)
    mdd e negativo (ex: -0.35 = -35%)
    """

    equity_value = equity.dropna().astype(float)
    if equity_value.empty:
        return float("nan"), None, None
    
    running_max = equity_value.cummax()

    drawdon = equity_value / running_max - 1.0

    max_drawdon_value = drawdon.min()

    end = drawdon.idxmin() if not np.isnan(max_drawdon_value) else None

    if end is None:
        return float("nan"), None, None
    
    #start = ponto do pico antes do vale

    start = equity_value.loc[:end].idxmax()
    return float (max_drawdon_value), start, end


def cagr(equity: pd.Series, periodos_por_ano: int = 8760) -> float:
    """
    CAGR aproximado usando contagem de períodos.
    """

    equity_value=equity.dropna().astype(float)
    if len(equity_value)<2:
        return float("nan")
    
    start_value = float(equity_value.iloc[0])
    end_value = float(equity_value.iloc[-1])

    if start_value <= 0:
        return float("nan")
    
    num_periods = len(equity_value) - 1
    years = num_periods / periodos_por_ano

    if years <= 0:
        return float("nan")
    
    return (end_value / start_value) ** (1.0/years)-1.0

def compute_metrics(
        equity: pd.Series,
        returns: pd.Series,
        periodos_por_ano: int = 8760,
) -> Metrics:
    s =  shape_ratio(returns, periodos_por_ano=periodos_por_ano)
    g = cagr(equity, periodos_por_ano=periodos_por_ano)

    mdd, start, end = max_drawdon(equity)

    return Metrics(shape=s, cagr=g, max_drawdown=mdd, max_drawdown_start=start, max_drawdown_end=end)

if __name__ == "__main__":
    # Demo: SMA vs Buy & Hold com seus módulos atuais
    from src.data.datasets import load_features
    from src.strategies.sma_cross import sma_crossover_signals
    from src.backtest.engine import run_backtest_long_only

    df = load_features("BTCUSDT", "1h")
    df_sig = sma_crossover_signals(df, 20, 100)

    # Estratégia
    res_nocost = run_backtest_long_only(df_sig, initial_capital=10_000, fee_rate=0.0, slippage=0.0)
    res_cost   = run_backtest_long_only(df_sig, initial_capital=10_000, fee_rate=0.001, slippage=0.0002)

    met_nocost = compute_metrics(res_nocost.equity, res_nocost.returns)
    met_cost   = compute_metrics(res_cost.equity, res_cost.returns)

    # Buy & Hold
    bh_returns = df_sig["ret"].fillna(0.0)
    bh_equity = 10_000 * (1.0 + bh_returns).cumprod()
    met_bh = compute_metrics(bh_equity, bh_returns)

    print("=== SMA(20/100) SEM CUSTO ===")
    print(f"Sharpe: {met_nocost.shape:.3f}")
    print(f"CAGR:   {met_nocost.cagr:.2%}")
    print(f"MDD:    {met_nocost.max_drawdown:.2%}  ({met_nocost.max_drawdown_start} -> {met_nocost.max_drawdown_end})")

    print("\n=== SMA(20/100) COM CUSTO ===")
    print(f"Sharpe: {met_cost.shape:.3f}")
    print(f"CAGR:   {met_cost.cagr:.2%}")
    print(f"MDD:    {met_cost.max_drawdown:.2%}  ({met_cost.max_drawdown_start} -> {met_cost.max_drawdown_end})")

    print("\n=== Buy & Hold ===")
    print(f"Sharpe: {met_bh.shape:.3f}")
    print(f"CAGR:   {met_bh.cagr:.2%}")
    print(f"MDD:    {met_bh.max_drawdown:.2%}  ({met_bh.max_drawdown_start} -> {met_bh.max_drawdown_end})")

