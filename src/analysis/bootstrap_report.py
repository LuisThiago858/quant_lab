from __future__ import annotations
from pathlib import Path
import pandas as pd

SYMBOL = "BTCUSDT"  #simbolo
TF="1h"     #tempo grafico
SHORT_W=20 #media curta
LONG_W=100 #media longa

in_csv = Path(f"data/processed/bootstrap_sma_{SYMBOL}_{TF}.csv")
out_txt = Path(f"data/processed/bootstrap_report_sma{SHORT_W}_{LONG_W}_{SYMBOL}_{TF}.txt")

def resumir_estatisticas(df_simulacoes: pd.DataFrame, coluna: str) -> dict:
    """Ëxtrai métricas de tendencia central e dispersão para o relatorio."""
    serie_dados = pd.to_numeric(df_simulacoes[coluna], errors="coerce").dropna()
    return {
        "media": float(serie_dados.mean()),
        "p05": float(serie_dados.quantile(0.05)),# limite inferior do Intervalo de Confiança
        "p50": float(serie_dados.quantile(0.50)),# mediana valor central
        "p95": float(serie_dados.quantile(0.95)),# limite superior do Intervalo de Confiança
        "minimo": float(serie_dados.min()),
        "maximo": float(serie_dados.max()),
        "total_amostras": int(serie_dados.shape[0]),
    }

def formatar_percentual(valor: float) -> str:
    return f"{valor: .2%}"

def formatar_decimal(valor: float) -> str:
    return f"{valor:,.3f}"

def main():
    if not in_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado:{in_csv}")
    
    df_resultados = pd.read_csv(in_csv)
    #calculo das metricas resumidas
    stats_sharpe = resumir_estatisticas(df_resultados, "sharpe")
    stats_cagr = resumir_estatisticas(df_resultados, "cagr")
    stats_mdd = resumir_estatisticas(df_resultados, "mdd")
    stats_patrimonio = resumir_estatisticas(df_resultados, "patrimonio_final") if "patrimonio_final" in df_resultados.columns else None

    linhas_relatorio = []
    linhas_relatorio.append(f"Relatorio de robustez [Bootstrap] - SMA({SHORT_W}/{LONG_W} {SYMBOL} {TF})")
    linhas_relatorio.append("-" * 80)

    linhas_relatorio.append("\n Interpretação em linguagem de asset management:")

    #CAGR retorno anualizado
    linhas_relatorio.append(
        f"- Retorno Anualizado (90% IC): Entre {formatar_percentual(stats_cagr['p05'])} e {formatar_percentual(stats_cagr['p95'])} "
        f"Mediana: {formatar_percentual(stats_cagr['p50'])}."
    )
    #Sharpe risco x retorno
    linhas_relatorio.append(
        f"- Índice Sharpe (90% IC): Entre {formatar_decimal(stats_sharpe['p05'])} e {formatar_decimal(stats_sharpe['p95'])}"
        f"Mediana: {formatar_decimal(stats_sharpe['p50'])}."
    )
    #maximo Drawdown risco de queda
    linhas_relatorio.append(
        f"- Drawdown maximoimo (90% IC): Entre {formatar_percentual(stats_mdd['p05'])} e {formatar_percentual(stats_mdd['p95'])}"
        f"Mediana: {formatar_percentual(stats_mdd['p50'])}"
    )

    if stats_patrimonio:
        linhas_relatorio.append(
            f"-Patrimônio Final Esperado: De {stats_patrimonio['p05']:,.2f} a {stats_patrimonio['p95']:,.2f} USD."
        )

    #Bloco mais técnico
    linhas_relatorio.append("\n" + "="*30 + "\n Resumo Estatístico Completo: \n" + "="*30)
    linhas_relatorio.append(f"Sharpe: media={formatar_decimal(stats_sharpe['media'])}  p05={formatar_decimal(stats_sharpe['p05'])}  p50={formatar_decimal(stats_sharpe['p50'])}  p95={formatar_decimal(stats_sharpe['p95'])}  minimo={formatar_decimal(stats_sharpe['minimo'])}  maximo={formatar_decimal(stats_sharpe['maximo'])}")
    linhas_relatorio.append(f"CAGR:   media={formatar_percentual(stats_cagr['media'])}  p05={formatar_percentual(stats_cagr['p05'])}  p50={formatar_percentual(stats_cagr['p50'])}  p95={formatar_percentual(stats_cagr['p95'])}  minimo={formatar_percentual(stats_cagr['minimo'])}  maximo={formatar_percentual(stats_cagr['maximo'])}")
    linhas_relatorio.append(f"MDD:    media={formatar_percentual(stats_mdd['media'])}  p05={formatar_percentual(stats_mdd['p05'])}  p50={formatar_percentual(stats_mdd['p50'])}  p95={formatar_percentual(stats_mdd['p95'])}  minimo={formatar_percentual(stats_mdd['minimo'])}  maximo={formatar_percentual(stats_mdd['maximo'])}")

    if stats_patrimonio is not None:
        linhas_relatorio.append(
            f"Final equity: media={stats_patrimonio['media']:,.2f}  p05={stats_patrimonio['p05']:,.2f}  p50={stats_patrimonio['p50']:,.2f}  p95={stats_patrimonio['p95']:,.2f}  minimo={stats_patrimonio['minimo']:,.2f}  maximo={stats_patrimonio['maximo']:,.2f}"
        )

    linhas_relatorio.append("")
    linhas_relatorio.append(f"Observações:")
    linhas_relatorio.append("- IC 90% = percentis 5% e 95% (p05/p95).")
    linhas_relatorio.append("- Isso NÃO é garantia de retorno; é apenas a distribuição do seu backtest sob reamostragem dos retornos.")
    linhas_relatorio.append("- Próximo upgrade profissional: block bootstrap (preserva regimes/volatilidade).")

    #salvar exibição
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(linhas_relatorio), encoding="utf-8")
    print("\n".join(linhas_relatorio))

if __name__ == "__main__":
    main()