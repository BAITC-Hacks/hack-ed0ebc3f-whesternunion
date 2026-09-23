import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wind_agent import api, service
from wind_agent.model import features


def make_archive(path, origins, horizon):
    rows = []
    for origin in origins:
        issued = origin - pd.Timedelta(hours=7)
        available = origin - pd.Timedelta(hours=1)
        for valid_time in pd.date_range(origin, periods=horizon, freq='h'):
            rows.append({'turbine': 1, 'issued_at': issued.isoformat(),
                         'available_at': available.isoformat(),
                         'valid_time': valid_time.isoformat(), 'wind_speed': 7.0,
                         'temperature': 3.0,
                         'source': f'fixture/model/{issued:%Y%m%d%H}',
                         'kind': 'forecast'})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_real_pipeline_bounds_and_temporal_split(dataset):
    result = service.run_forecast(persist=False)
    assert len(result['forecast']) == 48
    assert not result['weather']['eligible_for_competition']
    assert result['validation']['train_end'] < result['validation']['validation_start']
    assert result['validation']['validation_end'] < result['training_cutoff']
    for row in result['forecast']:
        assert 0 <= row['lower'] <= row['power'] <= row['upper'] <= 1


def test_future_observations_cannot_change_predictions(dataset):
    origin = '2026-01-20T00:00:00+05:00'
    first = service.run_forecast(origin=origin, persist=False)
    path = dataset / 'turbine_1.csv'
    frame = pd.read_csv(path)
    future = pd.to_datetime(frame.timestamp) >= pd.Timestamp('2026-01-20')
    frame.loc[future, ['wind_speed', 'power', 'temperature']] = [50, 1, -30]
    frame.to_csv(path, index=False)
    second = service.run_forecast(origin=origin, persist=False)
    assert first['forecast'] == second['forecast']


def test_feature_matrix_does_not_use_power():
    frame = pd.DataFrame({'wind_speed': [7], 'temperature': [3], 'power': [0]})
    before = features(frame)
    frame['power'] = 1
    pd.testing.assert_frame_equal(before, features(frame))


def test_backtest_without_actual_has_no_fabricated_metrics(dataset, weather_file):
    result = service.backtest(1, weather_file, '2026-02-01T00:00:00+05:00',
                              '2026-02-02T00:00:00+05:00', persist=False)
    assert result['total_predictions'] == 24
    assert result['scored_predictions'] == 0
    assert result['metrics'] is None
    assert all(row['actual'] is None for row in result['predictions'])


def test_backtest_keeps_full_48_hour_horizon_and_scores_only_february(dataset):
    start = pd.Timestamp('2026-01-31T00:00:00+05:00')
    score_start = pd.Timestamp('2026-02-01T00:00:00+05:00')
    end = pd.Timestamp('2026-02-02T00:00:00+05:00')
    archive = make_archive(dataset / 'weather.csv', [start, score_start], 48)
    result = service.backtest(1, archive, start, end, 48, persist=False,
                              score_start=score_start)
    assert len(result['runs']) == 2
    assert result['total_predictions'] == 96
    assert result['scored_predictions'] == 0
    assert result['metrics'] is None
    assert result['baseline_comparison'] is None
    assert result['predictions'][-1]['lead_hour'] == 47
    assert pd.Timestamp(result['predictions'][-1]['timestamp']) >= end
    assert all(row['actual'] is None for row in result['predictions'])


def test_backtest_compares_with_recent_power_only_on_matched_actuals(dataset):
    origin = pd.Timestamp('2026-01-20T00:00:00+05:00')
    archive = make_archive(dataset / 'weather.csv', [origin], 24)
    result = service.backtest(1, archive, origin, origin + pd.Timedelta(days=1),
                              24, persist=False)
    assert result['scored_predictions'] == 24
    assert result['baseline_comparison']['paired_predictions'] == 24
    assert result['baseline_comparison']['model']['mae'] >= 0
    assert result['baseline_comparison']['power_persistence_6h']['mae'] >= 0
    assert all(item['scored_predictions'] == 1 for item in result['metrics_by_lead'].values())


def test_score_end_is_independent_of_issue_end(dataset):
    origin = pd.Timestamp('2026-02-28T23:00:00+05:00')
    score_end = pd.Timestamp('2026-03-01T00:00:00+05:00')
    path = dataset / 'turbine_1.csv'
    measured = pd.read_csv(path)
    future = pd.DataFrame({'timestamp': pd.date_range('2026-02-28T23:00',
                           periods=49 * 6, freq='10min'), 'wind_speed': 9.0,
                           'power': 0.5, 'temperature': 4.0})
    pd.concat([measured, future]).to_csv(path, index=False)
    archive = make_archive(dataset / 'weather.csv', [origin], 48)
    result = service.backtest(1, archive, origin, origin + pd.Timedelta(days=1), 48,
                              persist=False, score_start=origin, score_end=score_end)
    assert result['total_predictions'] == 48
    assert result['scored_predictions'] == 1
    assert result['coverage']['missing_hours'] == 0
    assert result['predictions'][0]['actual'] == 0.5
    assert all(row['actual'] is None for row in result['predictions'][1:])


def test_february_actuals_score_without_entering_training(dataset, weather_file):
    origin = '2026-02-01T00:00:00+05:00'
    original = service.run_forecast(1, 24, origin, 'archive', weather_file, persist=False)
    path = dataset / 'turbine_1.csv'
    measured = pd.read_csv(path)
    future_hours = pd.date_range('2026-02-01', periods=24 * 6, freq='10min')
    future = pd.DataFrame({'timestamp': future_hours, 'wind_speed': 9.0,
                           'power': 0.5, 'temperature': 4.0})
    pd.concat([measured, future]).to_csv(path, index=False)
    result = service.backtest(1, weather_file, origin,
                              '2026-02-02T00:00:00+05:00', 24, persist=False)
    assert result['scored_predictions'] == 24
    assert result['metrics'] is not None
    assert [row['power'] for row in result['predictions']] == [
        row['power'] for row in original['forecast']]


def test_api_forecast_export_and_bad_requests(dataset, monkeypatch):
    monkeypatch.setattr(api, 'ARTIFACT_DIR', dataset / 'artifacts')
    client = TestClient(api.app)
    assert client.get('/').status_code == 200
    result = client.post('/api/forecast', json={'turbine': 2, 'horizon': 24})
    assert result.status_code == 200
    run = result.json()
    assert len(run['forecast']) == 24
    exported = client.get(f"/api/runs/{run['id']}/csv")
    assert exported.status_code == 200
    assert len(exported.text.strip().splitlines()) == 25
    assert client.get('/api/runs').json()[0]['id'] == run['id']
    assert client.post('/api/forecast', json={'horizon': 3}).status_code == 422
    assert client.post('/api/forecast', json={'origin': '2026-02-01'}).status_code == 422
    assert client.get('/api/runs/not-a-run').status_code == 404


def test_archive_default_origin_matches_downloader(dataset):
    origin = pd.Timestamp('2026-01-31T23:00:00+05:00')
    archive = make_archive(dataset / 'weather.csv', [origin], 48)
    result = service.run_forecast(1, 48, mode='archive', archive_path=archive, persist=False)
    assert pd.Timestamp(result['origin']) == origin
    assert len(result['forecast']) == 48


def test_api_uses_prepared_training_archive(dataset, monkeypatch):
    monkeypatch.setattr(api, 'ROOT', dataset)
    training_path = dataset / 'data' / 'weather_training.csv'
    training_path.parent.mkdir()
    training_path.write_text('fixture', encoding='utf-8')
    calls = []

    def fake_forecast(**kwargs):
        calls.append(kwargs)
        return {'id': 'fixture'}

    monkeypatch.setattr(api, 'run_forecast', fake_forecast)
    response = TestClient(api.app).post('/api/forecast', json={'mode': 'archive'})
    assert response.status_code == 200
    assert calls[0]['training_archive_path'] == training_path
