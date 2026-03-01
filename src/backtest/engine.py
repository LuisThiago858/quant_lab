from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
#
@dataclass(frozen=True)
class ResultadoBacktest:
    equity: pd.Series
    returns: pd.Series
    trades: int
    initial_capital: float

def run_backtest_long_only(
    df: pd.DataFrame,
    initial_capital: float = 10_000.00,
    position_col: str = "position",
    return_col: str = "ret",
    fee_rate: float = 0.001,      # 0.1%
    slippage: float = 0.0002,     # 0.02%
) -> ResultadoBacktest:
    """
    Backetest simples long only
        - position = 1 -> exposto ao retorno do ativo
        - position = 0 -> em caixa retorno 0
        - custo aplicado quando há troca de posição (entrada/saída):
          cost_per_side = fee_rate + slippage
    Assume execução no candle position já deve estar shiftado.
    """

    for col in (position_col, return_col):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")
        
    bt = df.copy()

    #garantias minimas
    bt = bt.sort_index()
    bt = bt[~bt.index.duplicated(keep="last")]

    # retorno bruto da estratégia (sem custo)
    bt["strategy_ret_gross"] = bt[position_col] * bt[return_col]

    # detecta trocas de posição (0->1 ou 1->0)
    bt["pos_change"] = bt[position_col].diff().abs().fillna(0).astype(int)

    cost_per_side = fee_rate + slippage
    bt["cost"] = bt["pos_change"] * cost_per_side

    # retorno líquido
    bt["strategy_ret_net"] = bt["strategy_ret_gross"] - bt["cost"]

    # equity
    bt["equity"] = initial_capital * (1.0 + bt["strategy_ret_net"]).cumprod()

    trades = int(bt["pos_change"].sum())

    return ResultadoBacktest(
        equity=bt["equity"],
        returns=bt["strategy_ret_net"],
        trades=trades,
        initial_capital=initial_capital,
    )

if __name__ == "__main__":
    import pandas as pd
    from src.data.datasets import load_features
    from src.strategies.sma_cross import sma_crossover_signals

    df = load_features("BTCUSDT", "1h")

    df_sig = sma_crossover_signals(df, short_window=20, long_window=100)

    result = run_backtest_long_only(df_sig, initial_capital=10_000, fee_rate=0.001, slippage=0.0002)

    print("Backtest concluído (COM custos)")
    print(f"Capital inicial: {result.initial_capital:,.2f}")
    print(f"Capital final:   {result.equity.iloc[-1]:,.2f}")
    print(f"Trades (trocas): {result.trades}")

    #preview
    print(df_sig[["close", "ret", "signal", "position"]].tail(5))
    print(result.equity.tail(5))