import pandas as pd
import pytest

from wind_agent.data import instant, load_history


def test_cutoff_excludes_incomplete_hour_and_future(dataset):
    history, report = load_history(1, '2026-01-02T00:30:00+05:00')
    assert len(history) == 24
    assert history.index.max() == pd.Timestamp('2026-01-01T18:00:00Z')
    assert report['usable_hours'] == 24


def test_missing_values_are_not_backfilled(dataset):
    path = dataset / 'turbine_1.csv'
    frame = pd.read_csv(path)
    frame.loc[:3, 'power'] = -1
    frame.to_csv(path, index=False)
    history, report = load_history(1)
    assert history.iloc[0].isna().all()
    assert report['invalid_rows'] == 4
    assert report['missing_hours'] == 1


def test_duplicates_do_not_inflate_coverage(dataset):
    path = dataset / 'turbine_1.csv'
    frame = pd.read_csv(path)
    frame = pd.concat([frame.iloc[4:], frame.iloc[[4, 4, 4, 4]]])
    frame.to_csv(path, index=False)
    history, report = load_history(1)
    assert report['duplicate_timestamps'] == 4
    assert history.iloc[0].isna().all()


def test_timezone_required():
    with pytest.raises(ValueError, match='часовой пояс'):
        instant('2026-02-01')
