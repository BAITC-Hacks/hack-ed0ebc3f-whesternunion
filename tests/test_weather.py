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


def test_duplicate_cannot_hide_behind_late_available_at(weather_file):
    frame = pd.read_csv(weather_file)
    duplicate = frame.iloc[[0]].copy()
    duplicate['available_at'] = '2026-02-01T00:00:00Z'
    pd.concat([frame, duplicate]).to_csv(weather_file, index=False)
    with pytest.raises(ValueError, match='дубликаты'):
        archive(weather_file, 1, ORIGIN, 48)


def test_archive_run_has_one_full_horizon_publication_time(weather_file):
    frame = pd.read_csv(weather_file)
    frame.loc[1, 'available_at'] = '2026-01-31T13:00:00Z'
    frame.to_csv(weather_file, index=False)
    with pytest.raises(ValueError, match='одно available_at'):
        archive(weather_file, 1, ORIGIN, 48)


@pytest.mark.parametrize('column,value', [('temperature', 280), ('wind_speed', 'bad'),
                                          ('source', ''), ('turbine', 3),
                                          ('valid_time', '2026-01-31T19:30:00Z'),
                                          ('issued_at', '2026-01-31T06:00:00')])
def test_contract_rejects_bad_units_metadata_and_times(weather_file, column, value):
    frame = pd.read_csv(weather_file)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    frame.to_csv(weather_file, index=False)
    with pytest.raises(ValueError):
        archive(weather_file, 1, ORIGIN, 48)


def provider_headers(**changes):
    return {'last-modified': 'Sat, 31 Jan 2026 15:59:20 GMT',
            'etag': '"' + 'a' * 32 + '"', 'x-amz-server-side-encryption': 'AES256', **changes}


def test_publication_is_provider_time_not_assumed_lag():
    from wind_agent.weather_download import publication_time
    published = publication_time(provider_headers(), '2026-01-31T12:00:00Z', '2026-01-31T18:00:00Z')
    assert published == pd.Timestamp('2026-01-31T15:59:20Z')


@pytest.mark.parametrize('changes', [
    {'etag': '"abc-3"'}, {'etag': ''}, {'last-modified': 'bad'},
    {'last-modified': 'Sat, 31 Jan 2026 18:01:00 GMT'},
    {'last-modified': 'Sat, 31 Jan 2026 11:59:00 GMT'},
    {'x-amz-server-side-encryption': 'aws:kms'},
])
def test_unconfirmed_or_late_publication_fails_closed(changes):
    from wind_agent.weather_download import publication_time
    with pytest.raises(ValueError):
        publication_time(provider_headers(**changes), '2026-01-31T12:00:00Z', '2026-01-31T18:00:00Z')


def test_index_requires_correct_fields_initialization_and_forecast_step():
    from wind_agent.weather_download import parse_index
    raw = (b'1:0:d=2026013112:TMP:2 m above ground:6 hour fcst:\n'
           b'2:100:d=2026013112:UGRD:100 m above ground:6 hour fcst:\n'
           b'3:200:d=2026013112:VGRD:100 m above ground:6 hour fcst:\n')
    issued = pd.Timestamp('2026-01-31T12:00:00Z')
    parsed = parse_index(raw, 300, issued, 6)
    assert parsed['temperature']['start'] == 0
    assert parsed['temperature']['end'] == 99
    assert parsed['v']['end'] == 299
    for invalid in [raw.replace(b'6 hour fcst', b'anl'), raw.replace(b'd=2026013112', b'd=2026013118'),
                    raw.replace(b'UGRD', b'TMP'), raw.replace(b'2:100', b'2:0')]:
        with pytest.raises(ValueError):
            parse_index(invalid, 300, issued, 6)


def test_http_range_cannot_silently_be_full_file_or_changed_object():
    import httpx
    from wind_agent.weather_download import checked_range
    blob = b'GRIB' + b'\x00' * 3 + b'\x02' + (20).to_bytes(8, 'big') + b'7777'
    headers = {'content-range': 'bytes 100-119/1000', 'etag': '"abc"'}
    assert checked_range(httpx.Response(206, content=blob, headers=headers), 100, 119, 1000, '"abc"') == blob
    for status, body, hdr in [(200, blob, headers), (206, blob[:-1], headers),
                              (206, blob, {**headers, 'etag': '"changed"'}),
                              (206, blob, {**headers, 'content-range': 'bytes 0-19/1000'})]:
        with pytest.raises(ValueError):
            checked_range(httpx.Response(status, content=body, headers=hdr), 100, 119, 1000, '"abc"')


def test_bilinear_interpolation_preserves_linear_field():
    from wind_agent.weather_download import interpolate
    points = [{'lat': lat, 'lon': lon, 'value': 2 * lat + 3 * lon}
              for lat in [43.5, 43.75] for lon in [78.5, 78.75]]
    value, evidence = interpolate(points, 43.64515, 78.535604)
    assert value == pytest.approx(2 * 43.64515 + 3 * 78.535604)
    assert sum(point['weight'] for point in evidence) == pytest.approx(1)
    with pytest.raises(ValueError):
        interpolate(points, 45., 78.535604)


def test_agreed_schedule_preserves_last_48_hours():
    from wind_agent.weather_download import schedule
    origins = schedule()
    assert len(origins) == 29
    assert origins[0] == pd.Timestamp('2026-01-31T18:00:00Z')
    assert origins[-1] == pd.Timestamp('2026-02-28T18:00:00Z')
    assert target_hours(origins[-1], 48)[-1] == pd.Timestamp('2026-03-02T17:00:00Z')


def test_temporary_http_failure_retries_but_missing_object_does_not(monkeypatch):
    import httpx
    from wind_agent import weather_download as loader
    monkeypatch.setattr(loader.time, 'sleep', lambda _: None)
    calls = []
    def transport(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200, content=b'ok')
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        assert loader.request(client, 'GET', 'https://example.com').content == b'ok'
    assert len(calls) == 2
    calls.clear()
    def missing(request):
        calls.append(request)
        return httpx.Response(404)
    with httpx.Client(transport=httpx.MockTransport(missing)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            loader.request(client, 'GET', 'https://example.com')
    assert len(calls) == 1


def test_failed_download_does_not_replace_previous_export(tmp_path, monkeypatch):
    from wind_agent import weather_download as loader
    output = tmp_path / 'weather.csv'
    output.write_text('previous successful artifact', encoding='utf-8')
    def unavailable(*args):
        raise ValueError('No confirmed publication')
    monkeypatch.setattr(loader, 'download_hour', unavailable)
    with pytest.raises(ValueError, match='No confirmed'):
        loader.fetch(output, tmp_path / 'raw', last=loader.FIRST_ORIGIN, workers=1)
    assert output.read_text(encoding='utf-8') == 'previous successful artifact'


def test_raw_cache_hash_mismatch_is_not_silently_used(tmp_path):
    from wind_agent import weather_download as loader
    import json
    issued = pd.Timestamp('2026-01-31T12:00:00Z')
    manifest = {'version': loader.VERSION, 'issued_at': issued.isoformat(), 'lead': 6,
                'locations': {'1': [43.64515, 78.535604]}, 'files': []}
    for filename in ['source.idx', 'temperature.grib2', 'u.grib2', 'v.grib2']:
        (tmp_path / filename).write_bytes(b'original')
        manifest['files'].append({'name': filename, 'size': 8, 'sha256': loader.sha256(b'original')})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'u.grib2').write_bytes(b'modified')
    with pytest.raises(ValueError, match='SHA-256'):
        loader.cached_hour(tmp_path, issued, 6, pd.Timestamp(ORIGIN), {1: (43.64515, 78.535604)})


def test_decoder_checks_original_grib_metadata_and_units():
    ec = pytest.importorskip('eccodes', reason='Install requirements-weather.txt for native GRIB verification')
    from wind_agent.weather_download import decode
    handle = ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
    try:
        for key, value in {
            'centre': 'kwbc', 'dataDate': 20260131, 'dataTime': 1200,
            'typeOfProcessedData': 1, 'productionStatusOfProcessedData': 0,
            'discipline': 0, 'parameterCategory': 0, 'parameterNumber': 0,
            'typeOfFirstFixedSurface': 103, 'scaleFactorOfFirstFixedSurface': 0,
            'scaledValueOfFirstFixedSurface': 2, 'forecastTime': 6,
            'Ni': 2, 'Nj': 2, 'latitudeOfFirstGridPointInDegrees': 43.75,
            'latitudeOfLastGridPointInDegrees': 43.5,
            'longitudeOfFirstGridPointInDegrees': 78.5,
            'longitudeOfLastGridPointInDegrees': 78.75,
            'iDirectionIncrementInDegrees': .25, 'jDirectionIncrementInDegrees': .25,
        }.items():
            ec.codes_set(handle, key, value)
        ec.codes_set_values(handle, [280., 282., 284., 286.])
        args = ('temperature', pd.Timestamp('2026-01-31T12:00:00Z'), 6, {1: (43.64515, 78.535604)})
        values, evidence = decode(ec.codes_get_message(handle), *args)
        assert 280 < values['1'] < 286
        assert evidence['units'] == 'K'
        # Analysis and non-operational reconstructions must not pass as archived forecasts.
        ec.codes_set(handle, 'typeOfProcessedData', 0)
        with pytest.raises(ValueError, match='metadata mismatch'):
            decode(ec.codes_get_message(handle), *args)
        ec.codes_set(handle, 'typeOfProcessedData', 1)
        ec.codes_set(handle, 'productionStatusOfProcessedData', 2)
        with pytest.raises(ValueError, match='metadata mismatch'):
            decode(ec.codes_get_message(handle), *args)
        ec.codes_set(handle, 'productionStatusOfProcessedData', 0)
        with pytest.raises(ValueError, match='metadata mismatch'):
            decode(ec.codes_get_message(handle), 'u', args[1], 6, args[3])
    finally:
        ec.codes_release(handle)
