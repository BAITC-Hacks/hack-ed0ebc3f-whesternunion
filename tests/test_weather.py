import pandas as pd
import pytest

from wind_agent.weather import archive, persistence, target_hours
from wind_agent.data import load_history

ORIGIN = '2026-01-31T19:00:00Z'


def test_full_archive_run(weather_file):
    frame, meta = archive(weather_file, 1, ORIGIN, 48)
    assert len(frame) == 48
    assert meta['eligible_for_competition']
    assert frame.index[0] == pd.Timestamp(ORIGIN)


@pytest.mark.parametrize('change', ['late_publication', 'future_issue', 'observation', 'hindcast', 'missing_hour', 'null_wind'])
def test_rejects_unsafe_weather(weather_file, change):
    frame = pd.read_csv(weather_file)
    if change == 'late_publication':
        frame['available_at'] = '2026-01-31T20:00:00Z'
    elif change == 'future_issue':
        frame['issued_at'] = '2026-01-31T20:00:00Z'
        frame['available_at'] = '2026-01-31T21:00:00Z'
    elif change in ['observation', 'hindcast']:
        frame['kind'] = change
    elif change == 'missing_hour':
        frame = frame.iloc[1:]
    else:
        frame.loc[0, 'wind_speed'] = None
    frame.to_csv(weather_file, index=False)
    with pytest.raises(ValueError):
        archive(weather_file, 1, ORIGIN, 48)


def test_falls_back_to_earlier_complete_run(weather_file):
    frame = pd.read_csv(weather_file)
    newer = frame.iloc[:24].copy()
    newer['issued_at'] = '2026-01-31T12:00:00Z'
    newer['available_at'] = '2026-01-31T18:00:00Z'
    newer['wind_speed'] = 12
    pd.concat([frame, newer]).to_csv(weather_file, index=False)
    weather, meta = archive(weather_file, 1, ORIGIN, 48)
    assert (weather.wind_speed == 7).all()
    assert meta['issued_at'] == '2026-01-31T06:00:00+00:00'


def test_rejects_stale_baseline(dataset):
    history, _ = load_history(1)
    with pytest.raises(ValueError, match='старше'):
        persistence(history, '2026-02-02T00:00:00Z', 24)


def test_hour_boundary_required():
    with pytest.raises(ValueError):
        target_hours('2026-01-31T19:30:00Z', 24)
