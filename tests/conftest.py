import numpy as np
import pandas as pd
import pytest

from wind_agent import data, service


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    stamps = pd.date_range('2026-01-01', periods=31 * 24 * 6, freq='10min')
    phase = np.arange(len(stamps))
    wind = 7 + 3 * np.sin(phase / 40)
    frame = pd.DataFrame({'timestamp': stamps, 'wind_speed': wind,
                          'power': np.clip((wind / 12) ** 3, 0, 1),
                          'temperature': 5 + np.cos(phase / 100)})
    for turbine in (1, 2):
        frame.to_csv(tmp_path / f'turbine_{turbine}.csv', index=False)
    monkeypatch.setattr(data, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(data, 'TIMEZONE', 'Etc/GMT-5')
    monkeypatch.setattr(service, 'ARTIFACT_DIR', tmp_path / 'artifacts')
    service.trained.cache_clear()
    service.archive_trained.cache_clear()
    yield tmp_path
    service.trained.cache_clear()
    service.archive_trained.cache_clear()


@pytest.fixture
def weather_file(tmp_path):
    hours = pd.date_range('2026-01-31T19:00:00Z', periods=48, freq='h')
    rows = pd.DataFrame({'turbine': 1, 'issued_at': '2026-01-31T06:00:00Z',
                         'available_at': '2026-01-31T12:00:00Z',
                         'valid_time': hours, 'wind_speed': 7.0, 'temperature': 3.0,
                         'source': 'test forecast fixture', 'kind': 'forecast'})
    path = tmp_path / 'weather.csv'
    rows.to_csv(path, index=False)
    return path
