from __future__ import annotations
import pandas as pd

#porue usar SMA crossover?
#Simplicidade: A estratégia é fácil de entender e implementar, tornando-a acessível para traders iniciantes.
#Identificação de Tendências: O cruzamento de médias móveis pode ajudar a identificar mudanças na direção do mercado, sinalizando potenciais pontos de entrada e saída.
#Versatilidade: Pode ser aplicada a diferentes mercados e prazos, permitindo queos traders adaptem a estratégia às suas preferências e objetivos de negociação.  

# O principal objetivo é identificar mudanças na direção do mercado, sinalizando potenciais pontos de entrada e saída.
#  A estratégia é baseada no cruzamento de duas médias móveis: uma de curto prazo (SMA curta) e outra de longo prazo (SMA longa).
#  Quando a SMA curta cruza acima da SMA longa, isso pode ser interpretado como um sinal de compra, indicando que o preço pode estar
#  entrando em uma tendência de alta. Por outro lado, quando a SMA curta cruza abaixo da SMA longa, isso pode ser visto como um sinal 
# de venda, sugerindo que o preço pode estar entrando em uma tendência de baixa. A simplicidade e a capacidade de identificar tendências 
# tornam a estratégia SMA Crossover uma escolha popular entre os traders.        
def sma_crossover_signals(
        df: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 100,
        price_col: str = "close",
) -> pd.DataFrame:
    """
    Gera sinais para estratégia SMA Crossover.

    Principais Regras
    signal = 1 se SMA(short) > SMA(long), senão 0, porque a posição só é confirmada no fechamento do candle atual
    position = signal.shift(1) evitar lookhead que opera no proximo candle porque o sinal só é confirmado no fechamento do candle atual

    Retorna um DataFrame com estas colunas
    sma_short: SMA de curto prazo
    sma_long: SMA de longo prazo
    signal
    position
    """
    if price_col not in df.columns:
        raise ValueError(f"Coluna de '{price_col}' não existe no dataframe.")
    
    if short_window <= 0 or long_window <=0:
        raise ValueError("As janelas precisam ser inteiros e > 0.")
    if short_window >= long_window:
        raise ValueError("short_window precisa ser menor que long_window.")
    

    out = df.copy()
    sma_s=f"sma_{short_window}"
    sma_l=f"sma_{long_window}"
    # o uso de rolling com short_window e long_window garante que as médias móveis sejam calculadas corretamente,
    # levando em consideração o número de períodos especificados para cada janela.
    # O parâmetro min_periods é definido como o mesmo valor da janela para garantir que a
    # média móvel só seja calculada quando houver dados suficientes disponíveis, evitando assim resultados distorcidos ou NaN nos primeiros períodos.
    out[sma_s]=out[price_col].rolling(short_window, min_periods=short_window).mean()
    out[sma_l]=out[price_col].rolling(long_window, min_periods=long_window).mean()

    #sinal "teórico" no mesmo candle porque o sinal só é confirmado no fechamento do candle atual
    out["signal"] = (out[sma_s] > out[sma_l]).astype(int)

    #posição que será usada no backtest (entra no candle seguinte porque o sinal só é confirmado no fechamento do candle atual)
    out["position"] = out["signal"].shift(1).fillna(0).astype(int)

    return out

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    path = Path("data/processed/BTCUSDT_1h_features.parquet")
    df = pd.read_parquet(path).sort_index()

    res = sma_crossover_signals(df, short_window=20, long_window=100)
    print(res[["close", "sma_20", "sma_100", "signal", "position"]].tail(15))
