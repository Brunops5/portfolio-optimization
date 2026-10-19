"""Coleta, cache persistente e preparação de dados de criptoativos."""

import os
import time
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yaml

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


BASE_URL = "https://api.exchange.coinbase.com"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
PROJECT_ROOT = CONFIG_PATH.parent


def get_cache_dir() -> Path:
    """Retorna o cache configurável, por padrão dentro da home do usuário."""
    configured = os.environ.get("PORTFOLIO_CACHE_DIR", CONFIG["paths"]["cache_dir"])
    return Path(configured).expanduser().resolve()


def fetch_candles(
    product_id: str,
    start: str,
    end: str,
    granularity: int = 86400,
    max_retries: int = 5,
) -> pd.DataFrame:
    endpoint = f"{BASE_URL}/products/{product_id}/candles"
    params = {"start": start, "end": end, "granularity": granularity}
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return pd.DataFrame()
                dataframe = pd.DataFrame(
                    data,
                    columns=["timestamp", "low", "high", "open", "close", "volume"],
                )
                dataframe["timestamp"] = pd.to_datetime(
                    dataframe["timestamp"], unit="s", utc=True
                ).dt.tz_localize(None)
                dataframe["source"] = "coinbase"
                return dataframe.sort_values("timestamp").reset_index(drop=True)
            if response.status_code == 404:
                # Produto inexistente na Coinbase: permite que o chamador use a
                # fonte secundária sem repetir uma requisição que não é transitória.
                return pd.DataFrame()
            if response.status_code == 429:
                wait_seconds = 2**attempt
                print(f"Rate limit em {product_id}; aguardando {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                print(f"Erro HTTP {response.status_code} para {product_id}")
                time.sleep(1)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Falha ao buscar {product_id}: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Falha ao buscar {product_id} após {max_retries} tentativas.")


def chunk_date_range(start: str, end: str, max_days: int = 299) -> List[tuple]:
    """Divide um intervalo inclusivo respeitando o limite de candles da Coinbase."""
    start_dt = pd.Timestamp(start).normalize()
    end_dt = pd.Timestamp(end).normalize()
    chunks = []
    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=max_days), end_dt)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_yfinance(asset: str, start: str, end: str) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance não está instalado para executar o fallback.")
    ticker = f"{asset}-USD"
    # O parâmetro end do Yahoo é exclusivo.
    exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=exclusive_end,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f"Yahoo Finance não retornou dados para {asset}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    dataframe = data[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    dataframe.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"]).dt.tz_localize(None)
    dataframe["source"] = "yahoo"
    return dataframe


def _download_interval(
    asset: str, start: str, end: str, allow_yahoo: bool = True
) -> pd.DataFrame:
    product_id = f"{asset}-USD"
    parts = []
    for chunk_start, chunk_end in chunk_date_range(start, end):
        part = fetch_candles(product_id, chunk_start, chunk_end)
        if not part.empty:
            parts.append(part)
        time.sleep(0.5)

    coinbase = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # Uma cobertura parcial no início do intervalo é esperada para ativos que
    # ainda não haviam sido lançados. Trocar essa série pelo Yahoo somente por
    # ela ser mais curta pode selecionar outro criptoativo com o mesmo ticker
    # (como ocorre com ARB). A Coinbase é, portanto, sempre a fonte primária
    # quando reconhece o produto.
    if not coinbase.empty:
        return coinbase

    if not allow_yahoo:
        print(
            f"[{asset}] Intervalo sem candles na Coinbase; "
            "mantendo a fonte já fixada no cache."
        )
        return pd.DataFrame()

    print(f"[{asset}] Produto sem dados na Coinbase; tentando Yahoo Finance...")
    try:
        return fetch_yfinance(asset, start, end)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            f"Nenhuma fonte retornou dados válidos para {asset}: {exc}"
        ) from exc


def _infer_legacy_source(columns: pd.Index) -> str:
    """Recupera a fonte pelo esquema gravado pelo carregador legado."""
    market_columns = [
        column for column in columns if column not in {"source", "source_status"}
    ]
    if market_columns[:4] == ["timestamp", "low", "high", "open"]:
        return "coinbase"
    if market_columns[:4] == ["timestamp", "open", "high", "low"]:
        return "yahoo"
    raise ValueError(
        "Cache legado sem esquema reconhecível; a fonte não pode ser inferida."
    )


def _normalise_market_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True).dt.tz_localize(None)
    if "source" not in result:
        result["source"] = "legacy"
    legacy = result["source"].eq("legacy")
    if legacy.any():
        inferred_source = _infer_legacy_source(result.columns)
        result.loc[legacy, "source"] = inferred_source
        if "source_status" not in result:
            result["source_status"] = "recorded"
        result.loc[legacy, "source_status"] = "inferred_from_legacy_schema"
    elif "source_status" not in result:
        result["source_status"] = "recorded"
    result["source_status"] = result["source_status"].fillna("recorded")
    return result.drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def _market_data_checksum(dataframe: pd.DataFrame) -> str:
    canonical = dataframe.sort_values("timestamp").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _provenance_record(
    asset: str,
    dataframe: pd.DataFrame,
    requested_start: str,
    requested_end: str,
) -> dict:
    timestamps = pd.DatetimeIndex(dataframe["timestamp"]).normalize()
    expected = pd.date_range(requested_start, requested_end, freq="D")
    return {
        "asset": asset,
        "product_id": f"{asset}-USD",
        "sources": "|".join(sorted(dataframe["source"].astype(str).unique())),
        "source_status": "|".join(
            sorted(dataframe["source_status"].astype(str).unique())
        ),
        "requested_start": pd.Timestamp(requested_start).date().isoformat(),
        "requested_end": pd.Timestamp(requested_end).date().isoformat(),
        "first_observation": timestamps.min().date().isoformat(),
        "last_observation": timestamps.max().date().isoformat(),
        "observations": len(dataframe),
        "missing_calendar_days": len(expected.difference(timestamps.unique())),
        "sha256": _market_data_checksum(dataframe),
    }


def _atomic_write_csv(dataframe: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    temporary.replace(destination)


def _missing_date_intervals(
    cached: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Agrupa em intervalos contíguos todos os dias ausentes do cache."""
    if cached.empty:
        return [(start_date, end_date)]

    covered = pd.DatetimeIndex(cached["timestamp"]).normalize().unique()
    missing = pd.date_range(start_date, end_date, freq="D").difference(covered)
    if missing.empty:
        return []

    groups = missing.to_series().groupby(
        missing.to_series().diff().ne(pd.Timedelta(days=1)).cumsum()
    )
    return [(group.iloc[0], group.iloc[-1]) for _, group in groups]


def download_asset(
    asset: str,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame:
    """Obtém dados usando cache persistente e baixa somente intervalos ausentes."""
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("A data final deve ser posterior à inicial.")

    cache_file = get_cache_dir() / "market-data" / f"{asset}-USD_86400.csv"
    cached = pd.DataFrame()
    if cache_file.exists() and not force:
        cached = _normalise_market_data(pd.read_csv(cache_file))
    elif not force:
        legacy_file = PROJECT_ROOT / "data" / "raw" / f"{asset}_daily.csv"
        if legacy_file.exists():
            cached = _normalise_market_data(pd.read_csv(legacy_file))

    if force or cached.empty:
        intervals = [(start_date, end_date)]
        if force:
            cached = pd.DataFrame()
    else:
        # A cobertura deriva de cada candle realmente armazenado, inclusive no
        # interior do intervalo. Metadados ou apenas as datas extremas nunca podem
        # transformar uma falha transitória em cobertura permanente.
        intervals = _missing_date_intervals(cached, start_date, end_date)

    downloaded = []
    # Uma vez que a Coinbase identificou o produto, todos os complementos do
    # cache permanecem nessa fonte. Isso evita unir séries de ativos homônimos
    # quando um intervalo anterior ao lançamento não possui candles.
    cache_uses_coinbase = (
        not cached.empty
        and "source" in cached
        and cached["source"].eq("coinbase").any()
    )
    for interval_start, interval_end in intervals:
        print(f"[{asset}] Baixando {interval_start.date()} a {interval_end.date()}...")
        try:
            downloaded.append(
                _download_interval(
                    asset,
                    str(interval_start.date()),
                    str(interval_end.date()),
                    allow_yahoo=not cache_uses_coinbase,
                )
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            if cached.empty:
                raise
            print(f"[{asset}] Não foi possível ampliar o cache ({exc}); usando dados existentes.")

    frames = [frame for frame in [cached, *downloaded] if not frame.empty]
    if not frames:
        raise ValueError(f"Nenhum dado disponível para {asset}.")
    complete = _normalise_market_data(pd.concat(frames, ignore_index=True))
    _atomic_write_csv(complete, cache_file)
    selected = complete.loc[
        complete["timestamp"].between(start_date, end_date, inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError(f"O cache de {asset} não cobre o intervalo solicitado.")
    print(f"[{asset}] {len(selected)} registros carregados de {cache_file}")
    return selected


def load_all_assets(
    assets: Optional[List[str]] = None,
    force: bool = False,
    return_provenance: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if assets is None:
        assets = CONFIG["assets"]
    close_prices = {}
    provenance = []
    failures = {}
    for asset in assets:
        try:
            dataframe = download_asset(
                asset,
                CONFIG["start_date"],
                CONFIG["end_date"],
                force=force,
            )
            close_prices[asset] = dataframe.set_index("timestamp")["close"]
            provenance.append(
                _provenance_record(
                    asset,
                    dataframe,
                    CONFIG["start_date"],
                    CONFIG["end_date"],
                )
            )
        except Exception as exc:
            failures[asset] = str(exc)
    if failures:
        details = "; ".join(f"{asset}: {message}" for asset, message in failures.items())
        raise RuntimeError(f"Falha ao carregar o universo completo ({details}).")
    prices = pd.DataFrame(close_prices).sort_index()
    if return_provenance:
        return prices, pd.DataFrame(provenance).sort_values("asset")
    return prices


def _fill_internal_gaps(
    prices: pd.DataFrame, max_gap_days: int
) -> pd.DataFrame:
    """Preenche apenas lacunas internas curtas de cada ativo."""
    if max_gap_days < 0:
        raise ValueError("max_gap_days não pode ser negativo.")
    filled = prices.copy().astype(float)
    for column in filled:
        valid = filled[column].dropna()
        if valid.empty:
            continue
        first, last = valid.index.min(), valid.index.max()
        if max_gap_days:
            filled.loc[first:last, column] = filled.loc[first:last, column].ffill(
                limit=max_gap_days
            )
    # Manter as linhas ausentes até pct_change impede que preços separados por
    # uma lacuna longa se tornem artificialmente consecutivos.
    return filled


def compute_returns(prices: pd.DataFrame, fill_method: str = "ffill") -> dict:
    """Calcula os preços tratados e retornos simples usados pelo experimento."""
    if fill_method != "ffill":
        raise ValueError("Somente fill_method='ffill' é suportado sem look-ahead.")
    max_gap_days = int(CONFIG["optimization"]["max_internal_gap_days"])
    prices_filled = _fill_internal_gaps(prices, max_gap_days=max_gap_days)
    daily_simple = prices_filled.pct_change(fill_method=None).dropna(how="all")
    return {
        "prices": prices_filled,
        "daily_simple": daily_simple,
    }


def save_processed_data(data: dict, outdir: Path, prefix: str = "nci") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for key, dataframe in data.items():
        path = outdir / f"{prefix}_{key}.csv"
        dataframe.to_csv(path)
        print(f"[SALVO] {path}")


def save_provenance_manifest(
    provenance: pd.DataFrame, outdir: Path, prefix: str = "nci"
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{prefix}_data_provenance.csv"
    provenance.to_csv(path, index=False)
    print(f"[SALVO] Manifesto de proveniência: {path}")


def main(
    force_download: bool = False,
) -> dict:
    print("=" * 60)
    print("FASE 1 — COLETA E PROCESSAMENTO DE DADOS")
    print("=" * 60)
    prices, provenance = load_all_assets(
        force=force_download,
        return_provenance=True,
    )
    data = compute_returns(prices)
    print(f"Período: {prices.index.min()} a {prices.index.max()}")
    print(f"Ativos carregados: {list(prices.columns)}")
    print(f"Retornos diários simples: {data['daily_simple'].shape}")
    save_processed_data(data, PROJECT_ROOT / CONFIG["paths"]["processed_data"])
    save_provenance_manifest(
        provenance,
        PROJECT_ROOT / CONFIG["paths"]["processed_data"],
    )
    return data


if __name__ == "__main__":
    main()
