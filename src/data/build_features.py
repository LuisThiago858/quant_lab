from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def load_raw(symbol: str, tf_label: str) -> pd.DataFrame:
    """"
    Carrega os dados brutos do parquet e retorna um DataFrame para o símbolo e tf_label especificados.
    """
    path = Path("data/raw")/f"{symbol}_{tf_label}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Não achei: {path}")
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("o dataset raw não está com DatetimeIndex.")
    
    #Ordena o DataFrame pelo índice (timestamp) para garantir que os dados
    #estejam em ordem cronológica, o que é crucial para análises de séries temporais
    # e backtesting. Isso é especialmente importante se os dados brutos não estiverem 
    # garantidos para estar em ordem, ou se houver duplicados que possam estar fora de ordem.
    # A ordenação garante que as operações subsequentes, como merges e cálculos de indicadores,
    # sejam feitas corretamente com base na sequência temporal dos dados.
    df = df.sort_index()
   
    df = df[~df.index.duplicated(keep="last")] #Remove duplicados, mantendo o último registro (mais recente) para cada timestamp
    return df

def add_basic_features(
        df:pd.DataFrame, #O DataFrame deve ter um índice de tempo e uma coluna de preço (definida por price_col) para calcular as features.
        price_col: str = "close", #A coluna de preço a ser usada para calcular as features. O padrão é "close", mas pode ser ajustado para "open", "high", "low" ou qualquer outra coluna de preço presente no DataFrame.
        vol_window: int = 24,#O número de períodos (por exemplo, horas) para calcular a volatilidade. O padrão é 24, o que significa que a volatilidade será calculada com base nos últimos 24 períodos (1 dia se os dados forem horários). Esse parâmetro pode ser ajustado para calcular a volatilidade em diferentes janelas de tempo, dependendo da frequência dos dados e do horizonte de previsão desejado.
        z_window: int = 24,#O número de períodos para calcular a média móvel e o desvio padrão usados no cálculo do Z-Score. O padrão é 24, o que significa que a média móvel e o desvio padrão serão calculados com base nos últimos 24 períodos. Esse parâmetro pode ser ajustado para usar diferentes janelas de tempo para o cálculo do Z-Score, dependendo da frequência dos dados e do horizonte de previsão desejado.
) -> pd.DataFrame:
    if price_col not in df.columns: #Verifica se a coluna de preço especificada existe no DataFrame. Se não existir, lança um erro informando que a coluna não foi encontrada e listando as colunas disponíveis no DataFrame para ajudar o usuário a identificar o problema.
        raise ValueError(f"Coluna '{price_col}' não encontrada no DataFrame.")
    
    out = df.copy()

    #retornos simples: (P_t - P_{t-1})-1, o que significa a variação percentual do preço de um período para o outro. É uma medida comum de retorno em finanças, que indica a porcentagem de ganho ou perda em relação ao preço anterior.
    out["ret"] = out[price_col].pct_change() #Calcula o retorno simples usando a função pct_change() do pandas, que calcula a variação percentual entre o preço atual e o preço anterior. O resultado é uma nova coluna "ret" que contém os retornos simples para cada período.

    #log return: ln(P_t / P_{t-1}), o que significa a diferença entre o logaritmo do preço atual e o logaritmo do preço anterior. O log return é frequentemente usado em finanças porque tem propriedades matemáticas que facilitam a análise, como a aditividade ao longo do tempo (os log returns podem ser somados para obter o log return total sobre um período) e a simetria em relação a ganhos e perdas.
    out["log_ret"] = np.log(out[price_col]).diff() #Calcula o log return usando a função diff() do pandas, que calcula a diferença entre o logaritmo do preço atual e o logaritmo do preço anterior. O resultado é uma nova coluna "log_ret" que contém os log returns para cada período.

    #volatilidade rolling do log_ret (desvio padrão) calculada com base em uma janela móvel de z_window períodos. A volatilidade é uma medida de dispersão dos retornos e é frequentemente usada para avaliar o risco de um ativo. O desvio padrão é uma medida comum de volatilidade, e a janela móvel permite calcular a volatilidade ao longo do tempo, refletindo as mudanças na variabilidade dos retornos.
    out[f"vol_{vol_window}"] = out["log_ret"].rolling(vol_window).std() #Calcula a volatilidade usando a função rolling() do pandas para criar uma janela móvel de z_window períodos e a função std() para calcular o desvio padrão dos log returns dentro dessa janela. O resultado é uma nova coluna "vol_{vol_window}" que contém a volatilidade calculada para cada período. 

    #z-score do retorno simples (opcional) 
    rolling_mean = out["ret"].rolling(z_window).mean() #Calcula a média móvel dos retornos simples usando a função rolling() do pandas para criar uma janela móvel de z_window períodos e a função mean() para calcular a média dos retornos dentro dessa janela. O resultado é uma nova série que contém a média móvel dos retornos para cada período.
    rolling_std = out["ret"].rolling(z_window).std() #Calcula o desvio padrão móvel dos retornos simples usando a função rolling() do pandas para criar uma janela móvel de z_window períodos e a função std() para calcular o desvio padrão dos retornos dentro dessa janela. O resultado é uma nova série que contém o desvio padrão móvel dos retornos para cada período.
    out[f"zret_{z_window}"] = (out["ret"]-rolling_mean)/rolling_std #Calcula o Z-Score dos retornos simples usando a fórmula (ret - média móvel) / desvio padrão móvel. O resultado é uma nova coluna "zret_{z_window}" que contém o Z-Score dos retornos para cada período. O Z-Score é uma medida de quão muitos desvios padrão um valor está acima ou abaixo da média, e pode ser usado para identificar retornos anormais ou extremos.

    return out

def save_processed(df: pd.DataFrame, symbol: str, tf_label: str) -> Path:
    #Salva o DataFrame processado em um arquivo parquet na pasta "data/processed" com o nome 
    # "{symbol}_{tf_label}_features.parquet". A função cria a pasta "data/processed"
    #  se ela não existir e retorna o caminho do arquivo salvo como um objeto Path.
    #  Isso permite que os dados processados sejam facilmente acessados para análises 
    # futuras ou para uso em backtests e modelos de machine learning.
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_{tf_label}_features.parquet"
    df.to_parquet(out_path, engine="pyarrow")

    return out_path

def main() -> None:
    symbol = "BTCUSDT"
    tf_label = "1h"

    df = load_raw(symbol, tf_label)
    df_feat = add_basic_features(df, vol_window=24, z_window=24)

    out_path = save_processed(df_feat, symbol, tf_label)

    print(f" Features salvas em: {out_path}")
    print(df_feat[["close", "ret", "log_ret", "vol_24", "zret_24"]].tail(10))


if __name__ == "__main__":
    main()