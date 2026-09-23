import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wind_agent import api, service
from wind_agent.model import features


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
