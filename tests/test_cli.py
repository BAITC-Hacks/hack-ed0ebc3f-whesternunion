import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from wind_agent import __main__ as cli


def test_default_backtest_runs_both_turbines_and_horizons(monkeypatch, capsys):
    calls = []
    def fake_backtest(turbine, archive, start, end, horizon, score_start):
        calls.append((turbine, horizon, start, score_start))
        return {'turbine': turbine, 'horizon': horizon, 'predictions': [], 'metrics': None}
    monkeypatch.setattr(cli, 'backtest', fake_backtest)
    monkeypatch.setattr(sys, 'argv', ['wind-agent', 'backtest', '--archive', 'weather.csv'])
    cli.main()
    assert [(turbine, horizon) for turbine, horizon, _, _ in calls] == [
        (1, 24), (1, 48), (2, 24), (2, 48)]
    assert all(start == '2026-01-31T00:00:00+05:00' for _, _, start, _ in calls)
    assert all(score_start == '2026-02-01T00:00:00+05:00'
               for _, _, _, score_start in calls)
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
