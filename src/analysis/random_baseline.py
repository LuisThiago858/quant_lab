from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.datasets import load_features
from src.strategies.sma_cross import sma_crossover_signals
from src.backtest.engine import run_backtest_long_only
from src.backtest.metrics import compute_metrics

SYMBOL = "BTCUSDT"
TF = "1h"

CAPITAL_INICIAL = 10_000
TAXA_CORRETAGEM = 0.001
DESVIO_EXECUCAO = 0.0002

SHORT_W = 20
LONG_W = 100

NUM_SIMULACOES_RANDOM = 300
SEED = 42

# Se True, o random tenta ter turnover parecido com o da SMA
CALIBRAR_POR_TURNOVER_SMA = True

# Usado apenas se a calibragem acima estiver desligada
PROBABILIDADE_TROCA_FIXA = 0.03


def calcular_turnover(df: pd.DataFrame, position_col: str = "position") -> int:
    """
    Conta quantas trocas de posição ocorreram (0->1 ou 1->0).
    """
    return int(df[position_col].diff().abs().fillna(0).sum())


def gerar_estrategia_aleatoria(
    df_dados: pd.DataFrame,
    prob_troca: float = 0.03,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Gera uma estratégia aleatória com turnover controlado.

    Lógica:
    - começa em 0 ou 1 aleatoriamente
    - em cada candle, normalmente mantém a posição
    - com probabilidade 'prob_troca', alterna a posição

    Isso é um benchmark mais justo do que sortear 0/1 em todo candle.
    """
    if not (0 <= prob_troca <= 1):
        raise ValueError("prob_troca deve estar entre 0 e 1.")

    rng = np.random.default_rng(seed)
    out = df_dados.copy()

    sinais = np.zeros(len(out), dtype=int)
    sinais[0] = rng.integers(0, 2)  # posição inicial aleatória

    for i in range(1, len(out)):
        if rng.random() < prob_troca:
            sinais[i] = 1 - sinais[i - 1]  # troca
        else:
            sinais[i] = sinais[i - 1]      # mantém

    out["signal"] = sinais
    out["position"] = pd.Series(out["signal"], index=out.index).shift(1).fillna(0).astype(int)

    return out


def avaliar_estrategia(df_com_posicao: pd.DataFrame, nome: str) -> dict:
    """
    Roda o motor de backtest e extrai métricas principais.
    """
    resultado = run_backtest_long_only(
        df_com_posicao,
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

    # -----------------------------
    # 1) SMA
    # -----------------------------
    df_sma = sma_crossover_signals(df, SHORT_W, LONG_W)
    resultado_sma = avaliar_estrategia(df_sma, f"SMA({SHORT_W}/{LONG_W})")
    turnover_sma = calcular_turnover(df_sma)

    # -----------------------------
    # 2) Buy & Hold
    # -----------------------------
    df_bh = df.copy()
    df_bh["signal"] = 1
    df_bh["position"] = 1
    resultado_bh = avaliar_estrategia(df_bh, "Buy & Hold")

    # -----------------------------
    # 3) Estratégia aleatória com turnover controlado
    # -----------------------------
    if CALIBRAR_POR_TURNOVER_SMA:
        prob_troca_random = turnover_sma / len(df)
    else:
        prob_troca_random = PROBABILIDADE_TROCA_FIXA

    resultados_random = []
    turnovers_random = []

    print(f"Iniciando {NUM_SIMULACOES_RANDOM} simulações aleatórias...")
    print(f"Turnover SMA: {turnover_sma}")
    print(f"Probabilidade de troca do random: {prob_troca_random:.4f}")

    for i in range(NUM_SIMULACOES_RANDOM):
        df_random = gerar_estrategia_aleatoria(
            df,
            prob_troca=prob_troca_random,
            seed=SEED + i,
        )

        turnovers_random.append(calcular_turnover(df_random))

        resultado = avaliar_estrategia(df_random, f"Random_{i}")
        resultados_random.append(resultado)

    df_random_resultados = pd.DataFrame(resultados_random)

    # -----------------------------
    # 4) Resumo estatístico do random
    # -----------------------------
    sharpe_random_media = df_random_resultados["sharpe"].mean()
    sharpe_random_p05 = df_random_resultados["sharpe"].quantile(0.05)
    sharpe_random_p95 = df_random_resultados["sharpe"].quantile(0.95)

    cagr_random_media = df_random_resultados["cagr"].mean()
    cagr_random_p05 = df_random_resultados["cagr"].quantile(0.05)
    cagr_random_p95 = df_random_resultados["cagr"].quantile(0.95)

    sma_sharpe_percentil = float((df_random_resultados["sharpe"] <= resultado_sma["sharpe"]).mean())
    sma_cagr_percentil = float((df_random_resultados["cagr"] <= resultado_sma["cagr"]).mean())

    # -----------------------------
    # 5) Prints
    # -----------------------------
    print("\n" + "=" * 60)
    print("Benchmarking: SMA vs Buy & Hold vs Random Strategy")
    print("=" * 60)

    print("\n[SMA]")
    print(f"Sharpe: {resultado_sma['sharpe']:.3f}")
    print(f"CAGR:   {resultado_sma['cagr']:.2%}")
    print(f"MDD:    {resultado_sma['mdd']:.2%}")
    print(f"Equity final: {resultado_sma['equity_final']:,.2f}")

    print("\n[Buy & Hold]")
    print(f"Sharpe: {resultado_bh['sharpe']:.3f}")
    print(f"CAGR:   {resultado_bh['cagr']:.2%}")
    print(f"MDD:    {resultado_bh['mdd']:.2%}")
    print(f"Equity final: {resultado_bh['equity_final']:,.2f}")

    print("\n[Random Strategy - distribuição]")
    print(f"Sharpe médio: {sharpe_random_media:.3f} | p05={sharpe_random_p05:.3f} | p95={sharpe_random_p95:.3f}")
    print(f"CAGR médio:   {cagr_random_media:.2%} | p05={cagr_random_p05:.2%} | p95={cagr_random_p95:.2%}")
    print(f"Turnover médio random: {np.mean(turnovers_random):.1f}")

    print("\n[Posição da SMA dentro da distribuição aleatória]")
    print(f"Sharpe da SMA acima de {sma_sharpe_percentil:.1%} das estratégias aleatórias")
    print(f"CAGR da SMA acima de {sma_cagr_percentil:.1%} das estratégias aleatórias")

    # -----------------------------
    # 6) Exportar
    # -----------------------------
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_random_resultados.to_csv(out_dir / "random_baseline_distribution.csv", index=False)

    resumo = pd.DataFrame([
        {
            "estrategia": resultado_sma["estrategia"],
            "sharpe": resultado_sma["sharpe"],
            "cagr": resultado_sma["cagr"],
            "mdd": resultado_sma["mdd"],
            "equity_final": resultado_sma["equity_final"],
            "turnover": turnover_sma,
        },
        {
            "estrategia": resultado_bh["estrategia"],
            "sharpe": resultado_bh["sharpe"],
            "cagr": resultado_bh["cagr"],
            "mdd": resultado_bh["mdd"],
            "equity_final": resultado_bh["equity_final"],
            "turnover": 0,
        },
    ])

    resumo.to_csv(out_dir / "benchmark_summary.csv", index=False)

    print("\n✅ Arquivos salvos em:")
    print(f"- {out_dir / 'random_baseline_distribution.csv'}")
    print(f"- {out_dir / 'benchmark_summary.csv'}")

    # -----------------------------
    # 7) Visualizações
    # -----------------------------
    plt.figure(figsize=(10, 5))
    df_random_resultados["sharpe"].hist(bins=40, alpha=0.7, color="gray")
    plt.axvline(resultado_sma["sharpe"], linestyle="--", linewidth=2, label="SMA")
    plt.axvline(resultado_bh["sharpe"], linestyle="--", linewidth=2, label="Buy & Hold")
    plt.title("Distribuição do Sharpe - Estratégias Aleatórias")
    plt.xlabel("Sharpe")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    df_random_resultados["cagr"].hist(bins=40, alpha=0.7, color="gray")
    plt.axvline(resultado_sma["cagr"], linestyle="--", linewidth=2, label="SMA")
    plt.axvline(resultado_bh["cagr"], linestyle="--", linewidth=2, label="Buy & Hold")
    plt.title("Distribuição do CAGR - Estratégias Aleatórias")
    plt.xlabel("CAGR")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()