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

def momentum_com_filtro_volatilidade(dados_precos: pd.DataFrame, janela_momentum: int = 24, coluna_volatilidade: str = "vol_24", percentil_corte: float = 0.50, modo_filtro: str="low", coluna_preco: str = "close") -> pd.DataFrame:
    """
    Estratégia de Momentum que só opera em determinados regimes de volatilidade.
    
    Lógica:
    1. Calcula se o preço subiu nos últimos 'janela_momentum' períodos.
    2. Define um limite de volatilidade baseado no percentil (ex: mediana 0.50).
    3. Só gera sinal de compra (1) se o momentum for positivo E a volatilidade 
       estiver no regime escolhido (abaixo ou acima do limite).
    """
    #Validacoes iniciais
    for col in [coluna_preco, coluna_volatilidade]:
        if col not in dados_precos.columns:
            raise ValueError(f"A coluna '{col} é obrigatória e não foi encontrada")
        
    dados = dados_precos.copy()
    #Calculo do momentum (Variacao Percentual)
    col_momentum = f"momentum_{janela_momentum}"
    dados[col_momentum] = dados[coluna_preco].pct_change(janela_momentum)
    #definicao do regime de volatidade calculo do limite
    limite_volatalidade = dados[coluna_volatilidade].quantile(percentil_corte)
    #aplicacao do filtro de regime
    if modo_filtro == "low":
        #opera apeas em periodod de calmaria com volatilidade abaixodo limite
        dentro_do_regime = dados[coluna_volatilidade] < limite_volatalidade
    elif modo_filtro =="high":
        #opera apeans em periodos explosivos com volatilidade acima do limite
        dentro_do_regime = dados[coluna_volatilidade] > limite_volatalidade
    else:
        raise ValueError("O modo_filtro deve ser 'low' ou ''high'")  
    #Geracao so sinal combinado
    #condicao precos subindo momentum > 0  e volatilidade corrta
    dados["signal"] = (
        (dados[col_momentum]>0) &
        dentro_do_regime
    ).astype(int)
    #Execucao shift para evitar olhar o futuro
    dados["position"] = dados["signal"].shift(1).fillna(0).astype(int)
    return dados

def momentum_com_limiar_de_forca(dados_precos: pd.DataFrame, janela_lookback: int = 24, limiar_minimo:float=0.01, coluna_preco:str="close") ->pd.DataFrame:
    """
    Estratégia de Momentum com Filtro de Ruído.
    Só gera sinal de compra se o retorno acumulado for superior ao limiar 
    especificado, evitando "violinos" em mercados laterais.
    """
    if coluna_preco not in dados_precos.columns:
        raise ValueError(f"A coluna '{coluna_preco}' não foi encontrada no DataFrame")
    saida = dados_precos.copy()

    #calculo do retorno acumulado no periodo Momentum
    coluna_momentum = f"momentum_{janela_lookback}"
    saida[coluna_momentum] = saida[coluna_preco].pct_change(janela_lookback)
    #geracao do sinal 1 se ultrapassar o limiar, senao 0
    #O uso do limiar ajuda a filtrar oscilacoes pequenas e irrelevantes
    saida["signal"] = (saida[coluna_momentum] > limiar_minimo).astype(int)
    #aplicar o shift(1) para evitar lookhead 
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