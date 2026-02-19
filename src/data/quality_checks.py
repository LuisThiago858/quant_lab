from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_parquet(path: Path) -> pd.DataFrame:
    """
    Carrega um arquivo parquet e retorna um DataFrame.
    """
    dados_carregados_df = pd.read_parquet(path)
    if not isinstance(dados_carregados_df.index, pd.DatetimeIndex):
        raise ValueError("O parquet não está com DatetimeIndex no index.")
    return dados_carregados_df

def tf_label_to_timedelta(tf_label: str) -> pd.Timedelta:
    """
    tf_label exemplos: "1h", "15m", "1d", Porque o formato é mais simples e direto para os usuários, e é fácil de converter para Timedelta.
    """
    tf_label = tf_label.strip().lower()

    if tf_label.endswith("m"):
        #Força a conversão para inteiro, garantindo que tf_label seja algo como "15m" e não "15.5m"
        return pd.Timedelta(minutes=int(tf_label[:-1]))#Força a conversão para inteiro, garantindo que tf_label seja algo como "15m" e não "15.5m"
    if tf_label.endswith("h"):
        #Força a conversão para inteiro, garantindo que tf_label seja algo como "1h" e não "1.5h"
        return pd.Timedelta(hours=int(tf_label[:-1]))
    if tf_label.endswith("d"):
        #Força a conversão para inteiro, garantindo que tf_label seja algo como "1d" e não "1.5d"
        return pd.Timedelta(days=int(tf_label[:-1]))
    
    raise ValueError(f"tf_label invalido: {tf_label}. deve usar algo como '1h', '15m', '1d'.")


def quality_report(df: pd.DataFrame, tf_delta: pd.Timedelta) -> tuple[str, pd.Dataframe]:
    """
    """
    lines: list[str]=[]

    is_monotonic = df.index.is_monotonic_increasing
    lines.append(f"Index monotonic increasing: {is_monotonic}")

    if not is_monotonic:
        lines.append(" -> AVISO: índice fora de ordem (vai ordenar no merge/bactest)")
    
    #Lidando com duplicações pelo index
    dupli_count = int(df.index.duplicated().sum()) #Conta o número de duplicações no index, convertendo para int para evitar problemas de tipo
    lines.append(f"Duplicados no index: {dupli_count}")

    #Nulos
    nulls_total = int(df.isna().sum().sum())#Conta o número total de valores nulos no DataFrame, convertendo para int para evitar problemas de tipo
    lines.append(f"Total de valores nulos (todas colunas): {nulls_total}")

    #nulos por coluna (top 10)
    nulls_by_col = df.isna().sum().sort_values(ascending=False)
    top_nulls = nulls_by_col[nulls_by_col > 0 ].head(10)
    #Se não houver colunas com nulos, top_nulls estará vazio,
    #  então vamos lidar com isso no relatório para evitar confusão. Se top_nulls estiver vazio,
    #  podemos indicar que não há colunas com nulos.
    if len(top_nulls) == 0:
        lines.append("Nulos por coluna top: nenhum")
    else:
        lines.append("Nulos por coluna (top 10):")
        for col, n in top_nulls.items():
            lines.append(f" -{col}: {int(n)}")

    # Valores inválidos em "close" (<=0) - isso é importante porque o preço de fechamento não pode ser zero ou negativo,
    #  e isso pode indicar dados corrompidos ou erros de coleta. Se a coluna "close" estiver presente, vamos contar quantos 
    # registros têm valores inválidos e reportar isso.
    if "close" in df.columns:
        invalid_close = int((df["close"]<=0).sum())
        lines.append(f"Registros com close <= 0: {invalid_close}")
    else:
        lines.append("AVISO: coluna 'close' não encontrado (não dá pra checar close <=0).")

    # ---Gaps no tempo porque o index é DatetimeIndex, podemos calcular a diferença entre os timestamps consecutivos e contar quantos estão acima do tf_delta esperado. Isso indica onde temos buracos no tempo dos dados, o que pode ser problemático para análises e backtests.
    df_sorted = df.sort_index()
    diffs = df_sorted.index.to_series().diff()

    gap_mask = diffs > tf_delta
    gap_count = int(gap_mask.sum())

    lines.append(f"Gaps (diff > {tf_delta}):{gap_count}")

    #Construir tabela dos gaps (inicio, fim, tamamho) para os primeiros 10 gaps, para ajudar a identificar onde estão os buracos nos dados.
    gaps_df = pd.DataFrame({
        "prev_time": df_sorted.index.to_series().shift(1), #O timestamp anterior, que é o início do gap
        "curr_time": df_sorted.index.to_series(), #O timestamp atual, que é o fim do gap
        "diff": diffs, #A diferença entre o timestamp atual e o anterior, que é o tamanho do gap
    })
    gaps_df = gaps_df[gap_mask].copy() #Filtra apenas os gaps usando a máscara

    if not gaps_df.empty:
        # Top 15 maiores gaps
        gaps_df = gaps_df.sort_values("diff", ascendign = False)
        lines.append("Maiores gaps (top 15):")
        for _,row in gaps_df.head(15).iterrows():
            lines.append(
                f" - {row['prev_time']} -> {row['curr_time']} | gap={row['diff']}"
            )
    
    #Resumo geal do relatório com número de linhas, início e fim do index, para dar uma visão geral dos dados. Isso ajuda a entender o período coberto pelos dados e a quantidade de registros disponíveis.
    lines.append("")
    lines.append("Resumo:")
    lines.append(f"Linhas: {len(df_sorted):,}")
    lines.append(f"Início: {df_sorted.index.min()}")
    lines.append(f"Fim: {df_sorted.index.max()}")


    report_text = "\n".join(lines)
    return report_text, gaps_df

def save_report(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

def run_quality_checks(symbol: str, tf_label: str) -> None:
    parquet_path = Path("data/raw") / f"{symbol}_{tf_label}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Não achei o parquet: {parquet_path}")
    
    df = load_parquet(parquet_path)
    tf_delta = tf_label_to_timedelta(tf_label)

    report_text, gaps_df = quality_report(df, tf_delta)

    out_path = Path("data/processed") / f"quality_report_{symbol}_{tf_label}.txt"
    save_report(report_text, out_path)

    print(f"Relatório de qualidade salvo em: {out_path}")
    print(report_text)

    #salva gaps em csv para análise posterior
    if not gaps_df.empty:
        gaps_out = Path("data/processed") / f"gaps_{symbol}_{tf_label}.csv"
        gaps_df.to_csv(gaps_out, index=False)
        print(f" Gaps salvos tambem em: {gaps_out}")

if __name__ == "__main__":
    run_quality_checks("BTCUSDT", "1h")