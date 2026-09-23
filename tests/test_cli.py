import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from wind_agent import __main__ as cli


def test_default_backtest_runs_both_turbines_and_horizons(monkeypatch, capsys):
    calls = []
    def fake_backtest(turbine, archive, start, end, horizon, score_start, score_end,
                      training_archive_path):
        calls.append((turbine, horizon, start, score_start, score_end, end))
        return {'turbine': turbine, 'horizon': horizon, 'predictions': [], 'metrics': None}
    monkeypatch.setattr(cli, 'backtest', fake_backtest)
    monkeypatch.setattr(sys, 'argv', ['wind-agent', 'backtest', '--archive', 'weather.csv'])
    cli.main()
    assert [(call[0], call[1]) for call in calls] == [
        (1, 24), (1, 48), (2, 24), (2, 48)]
    assert all(call[2] == '2026-01-31T23:00:00+05:00' for call in calls)
    assert all(call[3] == '2026-02-01T00:00:00+05:00' for call in calls)
    assert all(call[4] == '2026-03-01T00:00:00+05:00' for call in calls)
    assert all(call[5] == '2026-03-01T23:00:00+05:00' for call in calls)
    assert len(json.loads(capsys.readouterr().out)['reports']) == 4


def test_watch_retries_transient_failure_and_deduplicates(monkeypatch, capsys):
    attempts = []
    saved = []
    delays = []
    def fake_forecast(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise httpx.ConnectError('temporary network failure')
        return {'id': str(len(attempts)), 'input_fingerprint': 'same-input',
                'summary': {}, 'ai': {'status': 'not_requested'}}
    def fake_sleep(delay):
        delays.append(delay)
        if len(delays) == 3:
            raise KeyboardInterrupt
    monkeypatch.setattr(cli, 'run_forecast', fake_forecast)
    monkeypatch.setattr(cli, 'save_result', saved.append)
    monkeypatch.setattr(cli.time, 'sleep', fake_sleep)
    args = SimpleNamespace(turbine=1, horizon=48, origin=None, mode='live',
                           archive=None, ai=False, interval=300)
    with pytest.raises(KeyboardInterrupt):
        cli.watch_forecasts(args)
    assert len(attempts) == 3
    assert len(saved) == 1
    assert delays == [60, 300, 300]
    assert 'Повтор через 60 с' in capsys.readouterr().err


def test_run_case_cli_passes_options_and_prints_summary(monkeypatch, capsys):
    calls = []

    def fake_case(**kwargs):
        calls.append(kwargs)
        return {'id': 'case-id', 'status': 'complete', 'cached': False,
                'evaluation_status': 'no_actuals', 'summary_path': 'case.md'}

    monkeypatch.setattr(cli, 'run_case', fake_case)
    monkeypatch.setattr(sys, 'argv', ['wind-agent', 'run-case', '--offline', '--training-days', '0'])
    cli.main()
    assert calls[0]['offline']
    assert calls[0]['training_days'] == 0
    assert calls[0]['first_origin'] == '2026-01-31T23:00:00+05:00'
    assert json.loads(capsys.readouterr().out)['summary_path'] == 'case.md'


def test_case_watch_retries_and_observes_again(monkeypatch, capsys):
    attempts, delays = [], []

    def fake_case(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise httpx.ConnectError('temporary NOAA failure')
        return {'id': 'case-id', 'status': 'complete', 'cached': True,
                'evaluation_status': 'no_actuals', 'summary_path': 'case.md'}

    def stop_after_two(delay):
        delays.append(delay)
        if len(delays) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, 'run_case', fake_case)
    monkeypatch.setattr(cli.time, 'sleep', stop_after_two)
    monkeypatch.setattr(sys, 'argv', ['wind-agent', 'run-case', '--watch', '--interval', '300'])
    cli.main()
    assert len(attempts) == 2
    assert delays == [60, 300]
    assert 'Повтор через 60 с' in capsys.readouterr().err
