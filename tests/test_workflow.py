import json

import pandas as pd
import pytest

from wind_agent import service, weather_download, workflow


@pytest.fixture
def case_setup(dataset, monkeypatch):
    monkeypatch.setattr(workflow, 'ARTIFACT_DIR', dataset / 'artifacts')
    calls = []

    def download(path, raw_dir, first, last, workers):
        calls.append('fetch')
        rows = []
        for origin in weather_download.schedule(first, last):
            for turbine in (1, 2):
                for timestamp in pd.date_range(origin, periods=48, freq='h'):
                    rows.append({'turbine': turbine,
                                 'issued_at': (origin - pd.Timedelta(hours=6)).isoformat(),
                                 'available_at': (origin - pd.Timedelta(hours=2)).isoformat(),
                                 'valid_time': timestamp.isoformat(), 'wind_speed': 7.0,
                                 'temperature': 3.0, 'source': 'fixture/model/run',
                                 'kind': 'forecast'})
        pd.DataFrame(rows).to_csv(path, index=False)

    def validate(path, origins):
        calls.append('validate')
        return weather_download.validate_export(path, origins)

    monkeypatch.setattr(weather_download, 'fetch', download)
    monkeypatch.setattr(weather_download, 'validate_bundle', validate)
    kwargs = {'archive_path': dataset / 'weather.csv', 'raw_dir': dataset / 'raw',
              'first_origin': '2026-01-31T23:00:00+05:00',
              'last_origin': '2026-01-31T23:00:00+05:00',
              'score_start': '2026-02-01T00:00:00+05:00',
              'score_end': '2026-02-01T23:00:00+05:00', 'training_days': 0}
    monkeypatch.setattr(workflow.importlib.util, 'find_spec', lambda name: object())
    return kwargs, calls


def test_case_downloads_replays_verifies_and_reuses(dataset, case_setup, monkeypatch):
    kwargs, calls = case_setup
    result = workflow.run_case(**kwargs)
    assert calls == ['validate', 'fetch', 'validate']
    assert result['status'] == 'complete'
    assert result['evaluation_status'] == 'no_actuals'
    assert len(result['reports']) == 4
    assert all(item['coverage']['missing_hours'] == 0 for item in result['reports'])
    assert all(item['metrics'] is None for item in result['reports'])
    assert workflow.outputs_unchanged(result)
    assert [entry['action'] for entry in result['trace']].count('verify_forecast') == 4

    def unexpected_replay(*args, **kwargs):
        pytest.fail('Unchanged inputs must not trigger another replay.')

    monkeypatch.setattr(service, 'backtest', unexpected_replay)
    cached = workflow.run_case(**kwargs, offline=True)
    assert cached['cached']
    assert cached['id'] == result['id']
    assert calls.count('fetch') == 1


def test_case_recalculates_on_changed_data_or_output(dataset, case_setup):
    kwargs, calls = case_setup
    first = workflow.run_case(**kwargs)
    path = dataset / 'turbine_1.csv'
    frame = pd.read_csv(path)
    frame.loc[:100, 'power'] = 0.0
    frame.to_csv(path, index=False)
    second = workflow.run_case(**kwargs, offline=True)
    assert not second['cached']
    assert second['input_fingerprint'] != first['input_fingerprint']
    csv_path = second['reports'][0]['artifacts']['csv']
    with open(csv_path, 'a', encoding='utf-8') as output:
        output.write('damaged\n')
    third = workflow.run_case(**kwargs, offline=True)
    assert not third['cached']
    assert third['input_fingerprint'] == second['input_fingerprint']
    assert third['id'] != second['id']
    assert calls.count('fetch') == 1
    (dataset / 'artifacts' / 'case-latest.json').write_text('broken', encoding='utf-8')
    fourth = workflow.run_case(**kwargs, offline=True)
    assert not fourth['cached']
    assert fourth['status'] == 'complete'


def test_case_stops_on_invalid_evidence(dataset, case_setup, monkeypatch):
    kwargs, calls = case_setup

    def invalid_bundle(*args, **kwargs):
        raise ValueError('Raw SHA256 mismatch')

    monkeypatch.setattr(weather_download, 'validate_bundle', invalid_bundle)
    with pytest.raises(ValueError, match='SHA256'):
        workflow.run_case(**kwargs)
    assert not calls
    journals = list((dataset / 'artifacts').glob('case-*.json'))
    assert len(journals) == 1
    report = json.loads(journals[0].read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['trace'][-1]['action'] == 'stop'
    assert not (dataset / 'artifacts' / 'case-latest.json').exists()


def test_case_offline_never_downloads_missing_weather(dataset, case_setup):
    kwargs, calls = case_setup
    with pytest.raises(ValueError, match='Offline'):
        workflow.run_case(**kwargs, offline=True)
    assert calls == ['validate']


def test_replay_guard_rejects_future_weather():
    origin = pd.Timestamp('2026-01-31T18:00:00Z')
    rows = [{'origin': origin.isoformat(), 'timestamp': target.isoformat(),
             'weather_issued_at': origin.isoformat(),
             'weather_available_at': (origin + pd.Timedelta(hours=1)).isoformat(),
             'training_cutoff': origin.isoformat(),
             'lower': 0.1, 'power': 0.2, 'upper': 0.3}
            for target in pd.date_range(origin, periods=24, freq='h')]
    with pytest.raises(ValueError, match='unavailable'):
        workflow.check_replay({'horizon': 24, 'predictions': rows,
                               'coverage': {'missing_hours': 0}}, [origin])
