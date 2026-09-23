import numpy as np
import pandas as pd
import hashlib

from wind_agent.data import load_history
from wind_agent.model import archived_training_samples, fit_archive_model
from wind_agent import service


def training_archive(path, history, origin):
    rows = []
    for position, (valid_time, observed) in enumerate(
            history.loc[history.index < origin].dropna().iterrows()):
        issued = valid_time - pd.Timedelta(hours=7 + position % 48)
        rows.append({'turbine': 1, 'issued_at': issued.isoformat(),
                     'available_at': (issued + pd.Timedelta(hours=4)).isoformat(),
                     'valid_time': valid_time.isoformat(),
                     'wind_speed': observed.wind_speed * 1.15 + 0.2,
                     'temperature': observed.temperature + 1,
                     'source': f'fixture/model/{issued:%Y%m%d%H}',
                     'kind': 'forecast'})
    issued = origin - pd.Timedelta(hours=7)
    for valid_time in pd.date_range(origin, periods=48, freq='h'):
        rows.append({'turbine': 1, 'issued_at': issued.isoformat(),
                     'available_at': (origin - pd.Timedelta(hours=1)).isoformat(),
                     'valid_time': valid_time.isoformat(), 'wind_speed': 8.0,
                     'temperature': 4.0,
                     'source': f'fixture/model/{issued:%Y%m%d%H}', 'kind': 'forecast'})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_archive_model_uses_only_available_paired_history(dataset):
    history, _ = load_history(1)
    origin = pd.Timestamp('2026-01-31T19:00:00Z')
    archive = training_archive(dataset / 'weather.csv', history, origin)
    source = 'fixture/model/2026013112'
    before = archived_training_samples(history, archive, 1, origin, source)
    fitted = fit_archive_model(history, archive, 1, origin, source)
    assert fitted is not None
    assert fitted[2]['scope'] == 'power_on_archived_forecast_weather'
    assert pd.Timestamp(fitted[2]['validation_end']) < origin
    late = pd.read_csv(archive)
    new_row = late.iloc[0].copy()
    new_row['issued_at'] = '2026-02-01T00:00:00Z'
    new_row['available_at'] = '2026-02-01T01:00:00Z'
    new_row['wind_speed'] = 75
    pd.concat([late, pd.DataFrame([new_row])]).to_csv(archive, index=False)
    after = archived_training_samples(history, archive, 1, origin, source)
    assert len(before) == len(after)
    assert np.allclose(before.wind_speed, after.wind_speed)


def test_archive_forecast_selects_forecast_weather_model(dataset):
    history, _ = load_history(1)
    origin = pd.Timestamp('2026-01-31T19:00:00Z')
    archive = training_archive(dataset / 'weather.csv', history, origin)
    result = service.run_forecast(1, 48, origin, 'archive', archive, persist=False)
    assert result['model'] == service.ARCHIVE_MODEL_VERSION
    assert result['validation']['scope'] == 'power_on_archived_forecast_weather'
    assert result['validation']['validation_end'] < result['training_cutoff']
    assert len(result['forecast']) == 48
    assert all(0 <= row['power'] <= 1 for row in result['forecast'])


def test_separate_training_archive_and_future_actuals_are_isolated(dataset):
    history, _ = load_history(1)
    origin = pd.Timestamp('2026-01-31T19:00:00Z')
    train_path = training_archive(dataset / 'training.csv', history, origin)
    frame = pd.read_csv(train_path)
    future = pd.to_datetime(frame.valid_time, utc=True) >= origin
    forecast_path = dataset / 'forecast.csv'
    frame.loc[future].to_csv(forecast_path, index=False)
    frame.loc[~future].to_csv(train_path, index=False)
    before = service.run_forecast(1, 48, origin, 'archive', forecast_path, persist=False,
                                  training_archive_path=train_path)
    assert before['model'] == service.ARCHIVE_MODEL_VERSION
    assert before['weather']['training_archive_sha256'] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    measured_path = dataset / 'turbine_1.csv'
    observed = pd.read_csv(measured_path)
    added = pd.DataFrame({'timestamp': pd.date_range('2026-02-01', periods=48 * 6, freq='10min'),
                          'wind_speed': 50.0, 'power': 1.0, 'temperature': -30.0})
    pd.concat([observed, added]).to_csv(measured_path, index=False)
    after = service.run_forecast(1, 48, origin, 'archive', forecast_path, persist=False,
                                 training_archive_path=train_path)
    assert before['forecast'] == after['forecast']
    assert before['validation'] == after['validation']
