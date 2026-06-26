"""
Módulo de coleta e preprocessamento de dados de criptoativos.
Fonte: Coinbase Exchange REST API (dados públicos de candles diários).
"""

import os
import time
import yaml
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# Carrega configuração global
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

BASE_URL = "https://api.exchange.coinbase.com"
RAW_DIR = Path(CONFIG["paths"]["raw_data"])
PROCESSED_DIR = Path(CONFIG["paths"]["processed_data"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_candles(
    product_id: str,
    start: str,
    end: str,
    granularity: int = 86400,  # 1 dia em segundos
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Busca candles históricos da Coinbase Exchange.

    Args:
        product_id: Par de trading (ex: "BTC-USD").
        start: Data de início no formato ISO 8601.
        end: Data de fim no formato ISO 8601.
        granularity: Intervalo em segundos (86400 = 1 dia).
        max_retries: Número máximo de tentativas em caso de falha.

    Returns:
        DataFrame com colunas: [timestamp, low, high, open, close, volume]
    """
    endpoint = f"{BASE_URL}/products/{product_id}/candles"
    params = {
        "start": start,
        "end": end,
        "granularity": granularity,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(
                    data,
                    columns=["timestamp", "low", "high", "open", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            elif response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limit atingido para {product_id}. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                print(f"Erro {response.status_code} ao buscar {product_id}: {response.text}")
                time.sleep(1)
        except Exception as e:
            print(f"Exceção ao buscar {product_id} (tentativa {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Falha ao buscar dados para {product_id} após {max_retries} tentativas.")


def chunk_date_range(start: str, end: str, max_days: int = 300) -> List[tuple]:
    """
    Divide o período em chunks de no máximo max_days dias,
    pois a API Coinbase limita o número de candles por requisição.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=max_days), end_dt)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end
    return chunks


def fetch_yfinance(asset: str, start: str, end: str) -> pd.DataFrame:
    """Fallback para obter dados históricos via Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance não instalado. Execute: pip install yfinance")
    ticker = f"{asset}-USD"
    print(f"[{asset}] Fallback para Yahoo Finance ({ticker})...")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"yfinance não retornou dados para {asset}.")
    # yfinance retorna MultiIndex ou simples; normaliza
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.reset_index()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def download_asset(asset: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """
    Faz download dos candles diários para um ativo e salva em CSV.
    Se o arquivo já existir e force=False, carrega do cache.
    Se a Coinbase retornar poucos dados, faz fallback para Yahoo Finance.
    """
    product_id = f"{asset}-USD"
    filepath = RAW_DIR / f"{asset}_daily.csv"

    if filepath.exists() and not force:
        print(f"[{asset}] Carregando do cache...")
        return pd.read_csv(filepath, parse_dates=["timestamp"])

    print(f"[{asset}] Baixando dados da Coinbase...")
    chunks = chunk_date_range(start, end, max_days=300)
    dfs = []
    for chunk_start, chunk_end in chunks:
        df_chunk = fetch_candles(product_id, chunk_start, chunk_end)
        if not df_chunk.empty:
            dfs.append(df_chunk)
        time.sleep(0.5)

    if not dfs:
        df = fetch_yfinance(asset, start, end)
    else:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        # Verifica cobertura mínima (+-80% do esperado)
        expected_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
        if len(df) < expected_days * 0.7:
            print(f"[{asset}] Dados da Coinbase insuficientes ({len(df)} registros). Tentando Yahoo Finance...")
            df_yf = fetch_yfinance(asset, start, end)
            # Prioriza Yahoo Finance se for mais completo
            if len(df_yf) >= len(df):
                df = df_yf
            else:
                print(f"[{asset}] Mantendo dados da Coinbase (mais completos que YF).")

    df.to_csv(filepath, index=False)
    print(f"[{asset}] {len(df)} registros salvos em {filepath}")
    return df


def load_all_assets(assets: Optional[List[str]] = None, force: bool = False) -> pd.DataFrame:
    """
    Carrega/baixa todos os ativos e retorna um DataFrame wide com preços de fechamento.
    Índice: timestamp, Colunas: ativos.
    """
    if assets is None:
        assets = CONFIG["assets"]

    start = CONFIG["start_date"]
    end = CONFIG["end_date"]

    close_prices = {}
    for asset in assets:
        try:
            df = download_asset(asset, start, end, force=force)
            s = df.set_index("timestamp")["close"]
            s.index = pd.to_datetime(s.index)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            close_prices[asset] = s
        except Exception as e:
            print(f"[AVISO] Não foi possível carregar {asset}: {e}")

    prices = pd.DataFrame(close_prices)
    prices = prices.sort_index()
    return prices


def compute_returns(prices: pd.DataFrame, fill_method: str = "ffill") -> dict:
    """
    Calcula retornos logarítmicos diários e mensais a partir de preços.

    Returns:
        Dict com DataFrames: 'daily', 'monthly', 'prices_filled'
    """
    prices_filled = prices.ffill().bfill().dropna(how="all")
    # Remove fins de semana/feriados onde todos estão NaN
    prices_filled = prices_filled.dropna(how="all")

    # Retornos logarítmicos diários
    daily_returns = np.log(prices_filled / prices_filled.shift(1)).dropna()

    # Retornos mensais (último dia útil de cada mês)
    monthly_prices = prices_filled.resample("ME").last()
    monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

    return {
        "prices_filled": prices_filled,
        "daily": daily_returns,
        "monthly": monthly_returns,
    }


def save_processed_data(data: dict, prefix: str = "nci") -> None:
    """Persiste os dados processados em CSV."""
    for key, df in data.items():
        path = PROCESSED_DIR / f"{prefix}_{key}_returns.csv"
        df.to_csv(path)
        print(f"[SALVO] {path}")


def main(force_download: bool = False) -> dict:
    """Pipeline completo de carga e preprocessamento."""
    print("=" * 60)
    print("FASE 1 — COLETA E PROCESSAMENTO DE DADOS")
    print("=" * 60)

    prices = load_all_assets(force=force_download)
    print(f"\nPeríodo: {prices.index.min()} a {prices.index.max()}")
    print(f"Ativos carregados: {list(prices.columns)}")
    print(f"Shape (dias): {prices.shape}")

    data = compute_returns(prices)
    print(f"Retornos diários shape: {data['daily'].shape}")
    print(f"Retornos mensais shape: {data['monthly'].shape}")

    save_processed_data(data)
    return data


if __name__ == "__main__":
    main()
