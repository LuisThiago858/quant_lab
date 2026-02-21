from __future__ import annotations

from pathlib import Path
import pandas as pd

# Retorna o caminho do dataset de features para um símbolo e timeframe específico, seguindo a convenção de nomeação definida.
def features_path(symbol: str, tf_label: str) -> Path:
    return Path("data/processed") / f"{symbol}_{tf_label}_features.parquet"

# Carrega o dataset de features para um símbolo e timeframe específico, realizando
#  verificações básicas para garantir que o dataset esteja no formato esperado.
def load_features(symbol: str, tf_label: str) -> pd.DataFrame:
    path = features_path(symbol, tf_label)
    if not path.exists():
        raise FileNotFoundError(
            f"Não achado o dataset de features: {path}\n"
            f"Rode: python -m src.data.build_features"
        )
    
    df = pd.read_parquet(path)

    # Verificação basica para o backtest funcionar, o index precisa ser DatetimeIndex, ordenado e sem duplicatas.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataset de features precisa estar com DatetimeIndex.")
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Verificação basica para o backtest funcionar, o dataset precisa conter as colunas: open, high, low, close, volume, ret e log_ret.
    required = {"open", "high", "low", "close", "volume", "ret", "log_ret"}
    missing = required - set(df.columns) # Verifica se tem as colunas obrigatorias, se faltar alguma, levanta um erro.

    if missing:
        raise ValueError(f"Faltando colunas obrigatorias no dataset: {missing}")
    
    vol_cols = [c for c in df.columns if c.startswith("vol_")]
    if not vol_cols:
        raise ValueError("Nenhuma coluna de volatilidade encontrada (esperado algo como vol_24)")

    return df

if __name__ == "__main__":
    df = load_features("BTCUSDT", "1h")
    print(df.tail(5)[["close", "ret", "log_ret"] + [c for c in df.columns if c.startswith("vol_")][:1]])