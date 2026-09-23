from pathlib import Path
import hashlib

import httpx
import numpy as np
import pandas as pd

from .config import coordinates
from .data import instant

ARCHIVE_COLUMNS = ['turbine', 'issued_at', 'available_at', 'valid_time',
                   'wind_speed', 'temperature', 'source', 'kind']


def validate_archive_frame(frame):
    """Validate the shared CSV contract BEFORE selecting eligible runs."""
    if not set(ARCHIVE_COLUMNS).issubset(frame.columns):
        raise ValueError('Архив погоды: нужны столбцы ' + ', '.join(ARCHIVE_COLUMNS))
    frame = frame[ARCHIVE_COLUMNS].copy()
    if frame.empty:
        raise ValueError('Архив погоды пуст.')
    for column in ['issued_at', 'available_at', 'valid_time']:
        frame[column] = frame[column].map(instant)
    if not frame.turbine.isin([1, 2]).all():
        raise ValueError('В архиве допустимы только турбины 1 и 2.')
    if frame.source.isna().any() or frame.source.astype(str).str.strip().eq('').any():
        raise ValueError('У каждой строки должен быть непустой источник прогноза.')
    if not frame.kind.eq('forecast').all():
        raise ValueError('Архив принимает только forecast, не observation/reanalysis/hindcast.')
    if (frame.available_at < frame.issued_at).any():
        raise ValueError('available_at не может быть раньше issued_at.')
    if (frame.valid_time <= frame.issued_at).any():
        raise ValueError('valid_time должен быть позже инициализации прогноза.')
    if not frame.valid_time.eq(frame.valid_time.dt.floor('h')).all():
        raise ValueError('valid_time должен быть на границе часа.')
    group = ['turbine', 'issued_at', 'source']
    if frame.duplicated(group + ['valid_time']).any():
        raise ValueError('В архивном запуске есть дубликаты valid_time.')
    if (frame.groupby(group).available_at.nunique() != 1).any():
        raise ValueError('У запуска должно быть одно available_at: время доступности полного горизонта.')
    for column in ['wind_speed', 'temperature']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    if (not np.isfinite(frame[['wind_speed', 'temperature']]).all().all()
            or not frame.wind_speed.between(0, 75).all()
            or not frame.temperature.between(-80, 65).all()):
        raise ValueError('Некорректная погода: нужны конечные значения в м/с и °C.')
    return frame


def target_hours(origin, horizon):
    stamp = instant(origin)
    if stamp != stamp.floor('h'):
        raise ValueError('Момент расчёта должен быть на границе часа.')
    if horizon not in (24, 48):
        raise ValueError('Горизонт должен быть 24 или 48 часов.')
    return pd.date_range(stamp, periods=horizon, freq='h')


def validate_weather(frame, hours):
    result = frame.reindex(hours)[['wind_speed', 'temperature']].apply(pd.to_numeric, errors='coerce')
    if (result.isna().any().any() or not np.isfinite(result).all().all()
            or not result.wind_speed.between(0, 75).all()
            or not result.temperature.between(-80, 65).all()):
        raise ValueError('Погода не покрывает весь горизонт или содержит некорректные значения.')
    return result


def persistence(history, origin, horizon):
    hours = target_hours(origin, horizon)
    clean = history.dropna()
    age = (instant(origin) - (clean.index[-1] + pd.Timedelta(hours=1))).total_seconds() / 3600
    if age > 6:
        raise ValueError('Последние наблюдения старше 6 часов. Нужен архивный или live-прогноз погоды.')
    recent = clean.tail(6)
    frame = pd.DataFrame({'wind_speed': recent.wind_speed.mean(),
                          'temperature': recent.temperature.mean()}, index=hours)
    return frame, {'source': 'persistence_baseline', 'eligible_for_competition': False,
                   'note': 'Инерционный baseline: погода последних 6 часов. Это не внешний прогноз.'}


def archive(path: Path, turbine, origin, horizon):
    frame = validate_archive_frame(pd.read_csv(path))
    frame = frame[frame.turbine == turbine].copy()
    eligible = frame[(frame.issued_at <= instant(origin))
                     & (frame.available_at <= instant(origin)) & (frame.kind == 'forecast')]
    hours = target_hours(origin, horizon)
    # Pick ONE complete run, not a blend of later runs or observations.
    for (issued, source), run in reversed(list(eligible.groupby(['issued_at', 'source'], sort=True))):
        if run.valid_time.duplicated().any():
            raise ValueError('В архивном запуске есть дубликаты valid_time.')
        if not hours.isin(run.valid_time).all():
            continue
        result = validate_weather(run.set_index('valid_time'), hours)
        return result, {'source': str(source), 'issued_at': issued.isoformat(),
                        'available_at': run.available_at.max().isoformat(),
                        'eligible_for_competition': True,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    raise ValueError('Нет полного архивного прогноза, доступного на момент расчёта. Подмена фактической погодой запрещена.')


def live(turbine, horizon):
    lat, lon = coordinates(turbine)
    with httpx.Client(timeout=30) as client:
        response = client.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lat, 'longitude': lon,
            'hourly': 'wind_speed_100m,temperature_2m', 'wind_speed_unit': 'ms',
            'timezone': 'UTC', 'forecast_days': 4,
        })
        response.raise_for_status()
    retrieved = pd.Timestamp.now(tz='UTC')
    origin = retrieved.ceil('h')
    hourly = response.json()['hourly']
    frame = pd.DataFrame({'wind_speed': hourly['wind_speed_100m'],
                          'temperature': hourly['temperature_2m']},
                         index=pd.to_datetime(hourly['time'], utc=True))
    result = validate_weather(frame, target_hours(origin, horizon))
    return result, {'source': 'Open-Meteo live', 'retrieved_at': retrieved.isoformat(),
                    'eligible_for_competition': False, 'latitude': lat, 'longitude': lon,
                    'note': 'Текущий прогноз; не используется для исторического бэктеста. Ветер на 100 м требует калибровки к высоте датчика.'}, origin
