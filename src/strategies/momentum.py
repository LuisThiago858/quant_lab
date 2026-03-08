from __future__ import annotations
import pandas as pd

SYMBOL="BTCUSDT"
TF="1h"

def momentum_com_filtro_sma(dados_precos: pd.DataFrame, janela_lookback: int=24, janela_sma: int=100, coluna_preco:str="close")->pd.DataFrame:
    if coluna_preco not in dados_precos.columns:
        raise ValueError(f"Coluna '{coluna_preco}' não encontrada")
    saida = dados_precos.copy()
    saida[f"momentum_{janela_lookback}"] = saida[coluna_preco].pct_change(janela_lookback)
    saida[f"sma_{janela_sma}"] = saida[coluna_preco].rolling(janela_sma).mean()

    saida["signal"] = (
        (saida[f"momentum_{janela_lookback}"] > 0) &
        (saida[coluna_preco] > saida[f"sma_{janela_sma}"])
    ).astype(int)

    saida["position"] = saida["signal"].shift(1).fillna(0).astype(int)
    return saida

def sinais_momentum_absoluto(dados_precos: pd.DataFrame, janela_lookback: int = 24, coluna_preco: str = "close") -> pd.DataFrame:
    """
    Estrategia de time-series momentum (trend following puro)
    Lógica:
    - Comparar o preço atual com o preço de 'n' periodos atras.
    - Se o retorno for positivo momentum>0, sinaliza compra (1)
    - Se o retorno for negativo ou zero, sinaliza saída (0)
    """

    #Começando com validações de integridade
    if coluna_preco not in dados_precos.columns:
        raise ValueError(f"Coluna '{coluna_preco}' não existe no dataframe")
    if janela_lookback <= 0:
        raise ValueError("A janela de lookback deve ser um número inteiro positivo")

    df_estrategia = dados_precos.copy()
    #Calculo de momentum - Varicao percentual no periodo
    #preco_atual / preco_atras -1
    nome_coluna_mom = f"momentum_{janela_lookback}"
    df_estrategia[nome_coluna_mom] = df_estrategia[coluna_preco].pct_change(janela_lookback)
    #Geração de sinais: 1 se subiu, 0 se caiu ou ficou estavel
    df_estrategia["signal"] = (df_estrategia[nome_coluna_mom]>0).astype(int)
    #Execução - A posicao e assumida no proximo candle (shift 1) para evitar viés de antecipaçao
    df_estrategia["position"] = df_estrategia["signal"].shift(1).fillna(0).astype(int)

    return df_estrategia

if __name__ == "__main__":
    from src.data.datasets import load_features

    df_historico = load_features(SYMBOL, TF)
    df_resultado = sinais_momentum_absoluto(df_historico, janela_lookback=24)

    print(f"\nExemplo de sinais gerados (Lookback: 24h):")
    cols_visualizacao = ["close", "momentum_24", "signal", "position"]
    print(df_resultado[cols_visualizacao].tail(15))