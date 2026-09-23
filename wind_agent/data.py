from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .config import DATA_DIR, TIMEZONE, TRAIN_END

ALIASES = {
    'Статистическое время': 'timestamp',
    'Средняя скорость ветра(m/s)': 'wind_speed',
    'Нормализованная активная мощность': 'power',
    'Средняя температура окружающей среды(°C)': 'temperature',
}
COLUMNS = ['wind_speed', 'power', 'temperature']


def instant(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Время должно содержать часовой пояс, например +05:00.')
    return stamp.tz_convert('UTC')


def training_end() -> pd.Timestamp:
    return pd.Timestamp(TRAIN_END, tz=TIMEZONE).tz_convert('UTC')


def dataset_path(turbine: int) -> Path:
    if turbine not in (1, 2):
        raise ValueError('Допустимы турбины 1 и 2.')
    canonical = DATA_DIR / f'turbine_{turbine}.csv'
    if canonical.exists():
        return canonical
    matches = sorted(DATA_DIR.glob(f'*turbine {turbine}.csv'))
    if len(matches) != 1:
        raise ValueError(f'Нужен один CSV для турбины {turbine} в DATA_DIR.')
    return matches[0]


def load_history(turbine: int, before=None) -> tuple[pd.DataFrame, dict]:
    path = dataset_path(turbine)
    raw = pd.read_csv(path).rename(columns=ALIASES)
    if not {'timestamp', *COLUMNS}.issubset(raw.columns):
        raise ValueError('CSV должен содержать timestamp, wind_speed, power, temperature.')
    stamps = pd.to_datetime(raw.timestamp, errors='coerce', format='mixed')
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
    raw['timestamp'] = stamps.dt.tz_convert('UTC')
    invalid_times = int(raw.timestamp.isna().sum())
    raw = raw.dropna(subset=['timestamp'])
    # Drop future raw samples BEFORE aggregation. Never backfill missing data.
    if before is not None:
        raw = raw[raw.timestamp < instant(before)]
    duplicates = int(raw.timestamp.duplicated().sum())
    raw = raw.drop_duplicates('timestamp', keep='first').sort_values('timestamp')
    for column in COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors='coerce')
    valid = (np.isfinite(raw[COLUMNS]).all(axis=1)
             & raw.wind_speed.between(0, 75) & raw.power.between(0, 1)
             & raw.temperature.between(-80, 65))
    invalid_values = int((~valid).sum())
    raw.loc[~valid, COLUMNS] = np.nan
    indexed = raw.set_index('timestamp')[COLUMNS]
    hourly = indexed.resample('1h').mean()
    counts = indexed.resample('1h').count().min(axis=1)
    # Source cadence is 10 minutes: require at least 4 of 6 samples per hour.
    hourly.loc[counts < 4, COLUMNS] = np.nan
    if before is not None:
        hourly = hourly[hourly.index + pd.Timedelta(hours=1) <= instant(before)]
    report = {
        'source': path.name, 'timezone_assumption': TIMEZONE,
        'rows_before_cutoff': len(raw), 'duplicate_timestamps': duplicates,
        'invalid_timestamps': invalid_times, 'invalid_rows': invalid_values,
        'hours': len(hourly), 'usable_hours': int(hourly.dropna().shape[0]),
        'missing_hours': int(hourly.isna().any(axis=1).sum()),
        'start': hourly.index.min().isoformat() if len(hourly) else None,
        'end': hourly.index.max().isoformat() if len(hourly) else None,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if hourly.dropna().empty:
        raise ValueError('До выбранного момента нет пригодных часовых наблюдений.')
    return hourly, report
