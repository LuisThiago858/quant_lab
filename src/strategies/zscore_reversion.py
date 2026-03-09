from __future__ import annotations
import pandas as pd

def sinais_reversao_zscore(dados_precos: pd.DataFrame, janela_zscore: int = 48, limiar_compra: float = -2.0, limiar_venda: float = 2.0, coluna_preco: str = "close") -> pd.DataFrame:
    """
    Estratégia de Reversão à Média baseada em Z-Score (Long-Only).
    Logica:
    -compra quando o preço está estatisticamente 'barato' (Z-Score < limiar_compra).
    -sai da posição quando o preço está estatisticamente 'caro' (Z-Score > limiar_venda).
    """
    if coluna_preco not in dados_precos.columns:
        raise ValueError(f"Coluna '{coluna_preco}' não encontrada.")
    if janela_zscore <= 1:
        raise ValueError("A janela estatística deve ser maior que 1.")
    df_analise = dados_precos.copy()
    #cálcula de média e desvio padrão móvel
    media_movel = df_analise[coluna_preco].rolling(janela_zscore).mean()
    desvio_padrao = df_analise[coluna_preco].rolling(janela_zscore).std()
    #cálcula do Z-Score (Preço - Média) / Desvio
    col_z = f"zscore_{janela_zscore}"
    df_analise[col_z] = (df_analise[coluna_preco] - media_movel) / desvio_padrao

    #logica de execução atualmente maquina de estados
    sinais = []
    atualmente_comprado = 0
    for valor_z in df_analise[col_z]:
        if pd.isna(valor_z):
            sinais.append(0)
            continue

        #gatilho de entrada com preço muito abaixo da média
        if atualmente_comprado == 0 and valor_z < limiar_compra:
            atualmente_comprado = 1
        #gatilho de saida com preço muito acima da média (alvo atingido)
        elif atualmente_comprado == 1 and valor_z > limiar_venda:
            atualmente_comprado = 0

        sinais.append(atualmente_comprado)

    df_analise["signal"] = sinais
    #shift(1) fundamental para evitar look-ahead bias (executar no próximo candle)
    df_analise["position"] = df_analise["signal"].shift(1).fillna(0).astype(int)

    return df_analise


if __name__ == "__main__":
    from src.data.datasets import load_features

    df_dados = load_features("BTCUSDT", "1h")
    df_resultado = sinais_reversao_zscore(df_dados, janela_zscore=48, limiar_compra=-2.0, limiar_venda=2.0)

    print("\nExemplo de sinais gerados - Mean Reversion (z-score)")
    print(df_resultado[[ "close", "zscore_48", "signal", "position" ]].tail(20))