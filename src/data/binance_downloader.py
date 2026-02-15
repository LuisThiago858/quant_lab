from __future__ import annotations
from pathlib import Path
import pandas as pd
from binance.client import Client

def download_klines(
    symbol: str, #exemplo: "BTCUSDT" para o par de negociação Bitcoin/USDT
    interval: str, #exemplo: Client.KLINE_INTERVAL_1HOUR para candles de 1 hora, ou Client.KLINE_INTERVAL_1DAY para candles diários porque a binance tem uma série de intervalos pré-definidos para os candles, e esses intervalos são representados por constantes na classe Client da biblioteca python-binance.
    start_str: str, #exemplo: "2 years ago UTC" para baixar dados dos últimos 2 anos, ou "2021-01-01 UTC" para baixar dados a partir de uma data específica. A binance tem uma série de formatos de string que podem ser usados para especificar o período de tempo para os dados históricos, e esses formatos são interpretados pela função get_historical_klines da biblioteca python-binance.
    end_str: str | None = None, #exemplo: "1 day ago UTC" para baixar dados até 1 dia atrás, ou "2023-01-01 UTC" para baixar dados até uma data específica. Se end_str for None, a função irá baixar dados até o momento atual. Assim como start_str, a binance tem uma série de formatos de string que podem ser usados para especificar o período de tempo para os dados históricos, e esses formatos são interpretados pela função get_historical_klines da biblioteca python-binance.
) -> pd.DataFrame:
    """
    Baixa candles (klines) da binance e devolve Dataframe limpo:
    - DatetimeIndex (open_time, UTC)
    - colunas númericas em float
    - sem duplicações
    - ordenado por tempo
    """

    client = Client() #como esses dados são publicos não precisa de API key e secret key da binance para obtelos

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
    )

    if not klines:
        raise RuntimeError("Nenhum dado retornado. Verifique symbol/interval/período.")
    
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    #explicando um pouco sobre essas colunas criadas pela binance para cada candle (kline):
    
    #o que e um candle? é um gráfico de velas que representa o movimento de preço de um ativo em
    # um determinado período de tempo. Cada vela (candle) tem informações sobre o preço de abertura, 
    # fechamento, máxima e mínima durante esse período, além do volume negociado.

    #open_time: horário de abertura do candle
    #open: preço de abertura do candle
    #high: preço mais alto do candle
    #low: preço mais baixo do candle
    #close: preço de fechamento do candle
    #volume: volume negociado durante o candle
    #close_time: horário de fechamento do candle
    #quote_asset_volume: volume negociado em termos da moeda de cotação (por exemplo, USDT)
    #num_trades: número de negociações durante o candle
    #taker_buy_base_asset_volume: volume comprado por takers (ordens de mercado) em termos da moeda base
    #taker_buy_quote_asset_volume: volume comprado por takers em termos da moeda de cotação
    #ignore: campo ignorado, geralmente é 0

    df = pd.DataFrame(klines, columns=cols)
    #tratamentos de dados para deixar o DataFrame mais limpo e fácil de trabalhar

    #as colunas de tempo (open_time e close_time) são originalmente timestamps em milissegundos,
    #  então tem que ser convertidas para datetime.
    df["open_time"] = pd.to_datetime(df["open_time"], unit = "ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit = "ms", utc=True)
    #essas colunas são originalmente strings, mas representam valores numéricos, então tem que ser convertidas para float.
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume"]
    
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #num_trades é um campo inteiro, mas pode conter valores inválidos,
    #  então usamos Int64 que é uma extensão do pandas para lidar com inteiros que podem conter NaN.
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")


    #indexando por open_time, removendo duplicações, ordenando e limpando
    #dados inválidos, por exemplo, candles com preço de fechamento zero ou negativo,
    #o que não faz sentido para um ativo financeiro.
    df = df.set_index("open_time")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    return df

def save_raw_parquet(df: pd.DataFrame, symbol: str, tf_label:str) -> Path:
    """
    Salva o DataFrame em formato parquet, em data/raw/<symbol>_<tf>.parquet
    """
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{"symbol"}_{tf_label}.parquet"
    df.to_parquet(out_path, engine="pyarrow")

    return out_path


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1HOUR
    tf_label = "1h"
    start_str = "2 years ago UTC"

    df = download_klines(symbol, interval, start_str)
    #eu recomendo usar o data wrangler extensão do VS Code para visualizar os dados baixados, para ter uma ideia melhor 
    # do que foi obtido e verificar se está tudo certo antes de salvar em parquet. O data wrangler
    # é uma ferramenta de visualização de dados que pode ser muito útil para explorar e entender os dados baixados.
    out = save_raw_parquet(df, symbol, tf_label)

    print(f" Salvo: {out} ({len(df):,} linhas)")
    print(df.head(3)[["open", "high", "low", "close", "volume"]])
    print(df.tail(3)[["open", "high", "low", "close", "volume"]])

