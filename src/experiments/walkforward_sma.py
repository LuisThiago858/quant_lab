from __future__ import annotations
import pandas as pd
from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

CONFIGS = [
    (10, 50),
    (20, 100),
    (50, 200),
    (30, 150),
    (5, 200),
]

FEE = 0.001
SLIP = 0.0002
INITIAL = 10_000

def eval_config(df: pd.DataFrame, short: int, long: int) -> dict:
    df_signal = sma_crossover_signals(df, short, long)
    result =  run_backtest_long_only(df_signal, initial_capital=INITIAL, fee_rate=FEE, slippage=SLIP)
    metrics = compute_metrics(result.equity, result.returns)

    return {
        "short": short,
        "long": long,
        "sharpe": metrics.shape,
        "cagr": metrics.cagr,
        "mdd": metrics.max_drawdown,
    }

def main():
    symbol, tf = "BTCUSDT", "1H"
    df_walkforward = load_features(symbol, tf)

    #split temporal (sem embaralhar)
    split_date = df_walkforward.index.min() + (df_walkforward.index.max() - df_walkforward.index.min()) * 0.70
    train =df_walkforward.loc[:split_date].copy()
    test = df_walkforward.loc[split_date:].copy()

    print(f"Treino: {train.index.min()} -> {train.index.max()} (n={len(train)})")
    print(f"Teste: {test.index.min()} -> {test.index.max()} (n={len(test)})\n")

    # escolhe melhor config NO TREINO
    train_results = [eval_config(train, s, l) for s, l in CONFIGS]
    train_df = pd.DataFrame(train_results).sort_values("sharpe", ascending=False)

    best = train_df.iloc[0].to_dict()
    best_s, best_l = int(best["short"]), int(best["long"])


    print("Ranking no TREINO (por Sharpe)")
    print(train_df.to_string(index=False))
    print(f"\n Melhor no treino: SMA({best_s}/{best_l}) (Sharpe={best['sharpe']:.3f})\n")

    # avalia a MESMA config no TESTE
    test_metrics = eval_config(test, best_s, best_l)

    print("=== Resultado fora da amostra (TESTE) ===")
    print(f"SMA({best_s}/{best_l})  Sharpe={test_metrics['sharpe']:.3f}  "
          f"CAGR={test_metrics['cagr']:.2%}  MDD={test_metrics['mdd']:.2%}")

    # 3) baseline no TESTE (buy & hold)
    df_test_sig = test.copy()
    bh_ret = df_test_sig["ret"].fillna(0.0) if "ret" in df_test_sig.columns else None
    if bh_ret is None:
        # se você não tiver ret no features, dá pra recomputar aqui,
        # mas no seu projeto já existe ret.
        pass

    bh_equity = INITIAL * (1.0 + bh_ret.fillna(0.0)).cumprod()
    bh_met = compute_metrics(bh_equity, bh_ret.fillna(0.0))

    print("\nBuy & Hold no TESTE")
    print(f"Sharpe={bh_met.shape:.3f}  CAGR={bh_met.cagr:.2%}  MDD={bh_met.max_drawdown:.2%}")

    # salva tudo
    out = pd.DataFrame([{
        "split_date": split_date,
        "best_short_train": best_s,
        "best_long_train": best_l,
        "train_sharpe_best": best["sharpe"],
        "test_sharpe_best": test_metrics["sharpe"],
        "test_cagr_best": test_metrics["cagr"],
        "test_mdd_best": test_metrics["mdd"],
        "test_bh_sharpe": bh_met.shape,
        "test_bh_cagr": bh_met.cagr,
        "test_bh_mdd": bh_met.max_drawdown,
    }])

    out.to_csv("data/processed/walkforward_summary.csv", index=False)
    print("\nSalvo em data/processed/walkforward_summary.csv")


if __name__ == "__main__":
    main()