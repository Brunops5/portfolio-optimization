'''Download, cache, and prepare daily cryptocurrency market data.'''

import hashlib
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yaml

import names

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

BASE_URL = 'https://api.exchange.coinbase.com'
CONFIG_PATH = Path(__file__).resolve().parents[1] / names.CONFIG_FILENAME
with CONFIG_PATH.open('r', encoding='utf-8') as config_file:
    CONFIG = yaml.safe_load(config_file)
PROJECT_ROOT = CONFIG_PATH.parent


def get_cache_dir() -> Path:
    '''Return the configured cache directory, expanded to an absolute path.'''
    default_dir = CONFIG[names.PATHS][names.CACHE_DIR]
    configured = os.environ.get(names.CACHE_ENV_VARIABLE, default_dir)
    return Path(configured).expanduser().resolve()


def fetch_candles(
    product_id: str,
    start: str,
    end: str,
    granularity: int = 86400,
    max_retries: int = 5,
) -> pd.DataFrame:
    '''Fetch one Coinbase candle interval with retries for transient errors.'''
    endpoint = f'{BASE_URL}/products/{product_id}/candles'
    params = {
        names.API_START: start,
        names.API_END: end,
        names.GRANULARITY: granularity,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return pd.DataFrame()

                dataframe = pd.DataFrame(
                    data,
                    columns=[
                        names.TIMESTAMP,
                        names.LOW,
                        names.HIGH,
                        names.OPEN,
                        names.CLOSE,
                        names.VOLUME,
                    ],
                )
                dataframe[names.TIMESTAMP] = pd.to_datetime(
                    dataframe[names.TIMESTAMP],
                    unit='s',
                    utc=True,
                ).dt.tz_localize(None)
                dataframe[names.SOURCE] = names.COINBASE
                return dataframe.sort_values(names.TIMESTAMP).reset_index(drop=True)

            if response.status_code == 404:
                # An unknown product can use the fallback source immediately.
                return pd.DataFrame()

            if response.status_code == 429:
                wait_seconds = 2**attempt
                print(f'Rate limit for {product_id}; waiting {wait_seconds} seconds...')
                time.sleep(wait_seconds)
            else:
                print(f'HTTP {response.status_code} for {product_id}')
                time.sleep(1)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f'Could not fetch {product_id}: {exc}') from exc
            time.sleep(2**attempt)

    raise RuntimeError(f'Could not fetch {product_id} after {max_retries} attempts.')


def chunk_date_range(
    start: str,
    end: str,
    max_days: int = 299,
) -> List[tuple]:
    '''Split an inclusive date range into non-overlapping API intervals.'''
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    chunks = []
    current = start_date

    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days), end_date)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)

    return chunks


def fetch_yfinance(asset: str, start: str, end: str) -> pd.DataFrame:
    '''Download daily Yahoo Finance data when Coinbase has no product data.'''
    if not YFINANCE_AVAILABLE:
        raise ImportError('yfinance is required for the Yahoo Finance fallback.')

    ticker = f'{asset}-USD'
    # Yahoo Finance treats the end date as exclusive.
    exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime('%Y-%m-%d'),
        end=exclusive_end,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f'Yahoo Finance returned no data for {asset}.')

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    dataframe = data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    dataframe.columns = [
        names.TIMESTAMP,
        names.OPEN,
        names.HIGH,
        names.LOW,
        names.CLOSE,
        names.VOLUME,
    ]
    dataframe[names.TIMESTAMP] = pd.to_datetime(
        dataframe[names.TIMESTAMP]
    ).dt.tz_localize(None)
    dataframe[names.SOURCE] = names.YAHOO
    return dataframe


def _download_interval(
    asset: str,
    start: str,
    end: str,
    allow_yahoo: bool = True,
) -> pd.DataFrame:
    '''Download one interval without mixing sources for the same product.'''
    product_id = f'{asset}-USD'
    parts = []

    for chunk_start, chunk_end in chunk_date_range(start, end):
        part = fetch_candles(product_id, chunk_start, chunk_end)
        if not part.empty:
            parts.append(part)
        time.sleep(0.5)

    if parts:
        coinbase = pd.concat(parts, ignore_index=True)
    else:
        coinbase = pd.DataFrame()

    # Partial Coinbase history can simply mean that the asset launched later.
    if not coinbase.empty:
        return coinbase

    if not allow_yahoo:
        print(
            f'[{asset}] No Coinbase candles in this interval; '
            'keeping the source already recorded in the cache.'
        )
        return pd.DataFrame()

    print(f'[{asset}] No Coinbase data; trying Yahoo Finance...')
    try:
        return fetch_yfinance(asset, start, end)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(f'No source returned valid data for {asset}: {exc}') from exc


def _infer_legacy_source(columns: pd.Index) -> str:
    '''Infer the legacy file source from its original column order.'''
    market_columns = [
        column
        for column in columns
        if column not in {names.SOURCE, names.SOURCE_STATUS}
    ]
    coinbase_order = [names.TIMESTAMP, names.LOW, names.HIGH, names.OPEN]
    yahoo_order = [names.TIMESTAMP, names.OPEN, names.HIGH, names.LOW]

    if market_columns[:4] == coinbase_order:
        return names.COINBASE
    if market_columns[:4] == yahoo_order:
        return names.YAHOO

    raise ValueError(
        'Legacy cache has an unrecognized schema; its source cannot be inferred.'
    )


def _normalise_market_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Normalize timestamps and source metadata, then remove duplicate rows.'''
    result = dataframe.copy()
    result[names.TIMESTAMP] = pd.to_datetime(
        result[names.TIMESTAMP],
        utc=True,
    ).dt.tz_localize(None)

    if names.SOURCE not in result:
        result[names.SOURCE] = names.LEGACY

    legacy = result[names.SOURCE].eq(names.LEGACY)
    if legacy.any():
        inferred_source = _infer_legacy_source(result.columns)
        result.loc[legacy, names.SOURCE] = inferred_source
        if names.SOURCE_STATUS not in result:
            result[names.SOURCE_STATUS] = names.RECORDED
        result.loc[legacy, names.SOURCE_STATUS] = names.INFERRED_FROM_LEGACY_SCHEMA
    elif names.SOURCE_STATUS not in result:
        result[names.SOURCE_STATUS] = names.RECORDED

    result[names.SOURCE_STATUS] = result[names.SOURCE_STATUS].fillna(names.RECORDED)
    result = result.drop_duplicates(names.TIMESTAMP, keep='last')
    return result.sort_values(names.TIMESTAMP)


def _market_data_checksum(dataframe: pd.DataFrame) -> str:
    '''Calculate a stable SHA-256 checksum for the selected market data.'''
    canonical = dataframe.sort_values(names.TIMESTAMP).to_csv(index=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _provenance_record(
    asset: str,
    dataframe: pd.DataFrame,
    requested_start: str,
    requested_end: str,
) -> dict:
    '''Summarize source, coverage, and checksum for one asset.'''
    timestamps = pd.DatetimeIndex(dataframe[names.TIMESTAMP]).normalize()
    expected = pd.date_range(requested_start, requested_end, freq='D')
    sources = sorted(dataframe[names.SOURCE].astype(str).unique())
    source_statuses = sorted(dataframe[names.SOURCE_STATUS].astype(str).unique())

    return {
        names.ASSET: asset,
        names.PRODUCT_ID: f'{asset}-USD',
        names.SOURCES: '|'.join(sources),
        names.SOURCE_STATUS: '|'.join(source_statuses),
        names.REQUESTED_START: pd.Timestamp(requested_start).date().isoformat(),
        names.REQUESTED_END: pd.Timestamp(requested_end).date().isoformat(),
        names.FIRST_OBSERVATION: timestamps.min().date().isoformat(),
        names.LAST_OBSERVATION: timestamps.max().date().isoformat(),
        names.OBSERVATIONS: len(dataframe),
        names.MISSING_CALENDAR_DAYS: len(expected.difference(timestamps.unique())),
        names.SHA256: _market_data_checksum(dataframe),
    }


def _atomic_write_csv(dataframe: pd.DataFrame, destination: Path) -> None:
    '''Write a temporary CSV and replace the destination only when complete.'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    dataframe.to_csv(temporary, index=False)
    temporary.replace(destination)


def _missing_date_intervals(
    cached: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    '''Group all missing cache dates into contiguous download intervals.'''
    if cached.empty:
        return [(start_date, end_date)]

    covered = pd.DatetimeIndex(cached[names.TIMESTAMP]).normalize().unique()
    expected = pd.date_range(start_date, end_date, freq='D')
    missing = expected.difference(covered)
    if missing.empty:
        return []

    missing_series = missing.to_series()
    new_group = missing_series.diff().ne(pd.Timedelta(days=1)).cumsum()
    groups = missing_series.groupby(new_group)
    return [(group.iloc[0], group.iloc[-1]) for _, group in groups]


def download_asset(
    asset: str,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame:
    '''Load cached candles and download only the missing date intervals.

    A failed extension never becomes cached coverage. If older cached data is
    available, it is returned and the missing interval is retried next time.
    '''
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError('The end date must be on or after the start date.')

    cache_file = get_cache_dir() / 'market-data' / f'{asset}-USD_86400.csv'
    cached = pd.DataFrame()
    if cache_file.exists() and not force:
        cached = _normalise_market_data(pd.read_csv(cache_file))
    elif not force:
        legacy_file = PROJECT_ROOT / 'data' / 'raw' / f'{asset}_daily.csv'
        if legacy_file.exists():
            cached = _normalise_market_data(pd.read_csv(legacy_file))

    if force:
        cached = pd.DataFrame()
        intervals = [(start_date, end_date)]
    elif cached.empty:
        intervals = [(start_date, end_date)]
    else:
        intervals = _missing_date_intervals(cached, start_date, end_date)

    downloaded = []
    cache_uses_coinbase = False
    if not cached.empty and names.SOURCE in cached:
        cache_uses_coinbase = cached[names.SOURCE].eq(names.COINBASE).any()

    for interval_start, interval_end in intervals:
        print(
            f'[{asset}] Downloading {interval_start.date()} to {interval_end.date()}...'
        )
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
            print(f'[{asset}] Could not extend the cache ({exc}); using existing data.')

    frames = [frame for frame in [cached, *downloaded] if not frame.empty]
    if not frames:
        raise ValueError(f'No data available for {asset}.')

    complete = _normalise_market_data(pd.concat(frames, ignore_index=True))
    _atomic_write_csv(complete, cache_file)
    selected = complete.loc[
        complete[names.TIMESTAMP].between(
            start_date,
            end_date,
            inclusive='both',
        )
    ].copy()
    if selected.empty:
        raise ValueError(f'The {asset} cache does not cover the requested period.')

    print(f'[{asset}] Loaded {len(selected)} rows from {cache_file}')
    return selected


def load_all_assets(
    assets: Optional[List[str]] = None,
    force: bool = False,
    return_provenance: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    '''Load every configured asset and fail if any series is unavailable.'''
    if assets is None:
        assets = CONFIG[names.ASSETS]

    close_prices = {}
    provenance = []
    failures = {}
    for asset in assets:
        try:
            dataframe = download_asset(
                asset,
                CONFIG[names.START_DATE],
                CONFIG[names.END_DATE],
                force=force,
            )
            close_prices[asset] = dataframe.set_index(names.TIMESTAMP)[names.CLOSE]
            provenance.append(
                _provenance_record(
                    asset,
                    dataframe,
                    CONFIG[names.START_DATE],
                    CONFIG[names.END_DATE],
                )
            )
        except Exception as exc:
            failures[asset] = str(exc)

    if failures:
        details = '; '.join(
            f'{asset}: {message}' for asset, message in failures.items()
        )
        raise RuntimeError(f'Could not load the complete universe ({details}).')

    prices = pd.DataFrame(close_prices).sort_index()
    if return_provenance:
        provenance_df = pd.DataFrame(provenance).sort_values(names.ASSET)
        return prices, provenance_df
    return prices


def _fill_internal_gaps(
    prices: pd.DataFrame,
    max_gap_days: int,
) -> pd.DataFrame:
    '''Forward-fill only short gaps between the first and last asset price.'''
    if max_gap_days < 0:
        raise ValueError('max_gap_days cannot be negative.')

    filled = prices.copy().astype(float)
    for column in filled:
        valid = filled[column].dropna()
        if valid.empty:
            continue

        first = valid.index.min()
        last = valid.index.max()
        if max_gap_days:
            filled.loc[first:last, column] = filled.loc[
                first:last,
                column,
            ].ffill(limit=max_gap_days)

    # Long gaps stay missing so pct_change cannot join unrelated price periods.
    return filled


def compute_returns(
    prices: pd.DataFrame,
    fill_method: str = names.FORWARD_FILL,
) -> dict:
    '''Return cleaned prices and simple daily returns without backward fill.'''
    if fill_method != names.FORWARD_FILL:
        raise ValueError('Only the ffill fill_method is supported without look-ahead.')

    max_gap_days = int(CONFIG[names.OPTIMIZATION][names.MAX_INTERNAL_GAP_DAYS])
    prices_filled = _fill_internal_gaps(prices, max_gap_days=max_gap_days)
    daily_simple = prices_filled.pct_change(fill_method=None).dropna(how='all')
    return {
        names.PRICES: prices_filled,
        names.DAILY_SIMPLE: daily_simple,
    }


def save_processed_data(
    data: dict,
    outdir: Path,
    prefix: str = 'nci',
) -> None:
    '''Save cleaned prices and returns as separate CSV files.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for key, dataframe in data.items():
        path = outdir / f'{prefix}_{key}.csv'
        dataframe.to_csv(path)
        print(f'[SAVED] {path}')


def save_provenance_manifest(
    provenance: pd.DataFrame,
    outdir: Path,
    prefix: str = 'nci',
) -> None:
    '''Save the data source and coverage manifest for the current run.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{prefix}_data_provenance.csv'
    provenance.to_csv(path, index=False)
    print(f'[SAVED] Provenance manifest: {path}')


def main(force_download: bool = False) -> dict:
    '''Load, clean, save, and return the data used by the experiment.'''
    print('=' * 60)
    print('STEP 1 — DATA COLLECTION AND PROCESSING')
    print('=' * 60)

    prices, provenance = load_all_assets(
        force=force_download,
        return_provenance=True,
    )
    data = compute_returns(prices)

    print(f'Period: {prices.index.min()} to {prices.index.max()}')
    print(f'Assets loaded: {list(prices.columns)}')
    print(f'Daily simple returns: {data[names.DAILY_SIMPLE].shape}')

    processed_dir = (
        PROJECT_ROOT / Path(CONFIG[names.PATHS][names.PROCESSED_DATA]).expanduser()
    )
    save_processed_data(data, processed_dir)
    save_provenance_manifest(provenance, processed_dir)
    return data


if __name__ == '__main__':
    main()
