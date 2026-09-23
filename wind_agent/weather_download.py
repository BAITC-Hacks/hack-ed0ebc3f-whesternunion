"""Download original operational NOAA GFS messages with publication evidence.

Run independently of the model/replay CLI: python -m wind_agent.weather_download.
Only original GRIB byte ranges are stored, never a reconstructed weather product.
"""
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid

import httpx
import pandas as pd

from .config import COORDINATES, ROOT
from .data import instant
from .weather import ARCHIVE_COLUMNS, archive, validate_archive_frame

BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
FIRST_ORIGIN = '2026-01-31T23:00:00+05:00'
LAST_ORIGIN = '2026-02-28T23:00:00+05:00'
VERSION = 'noaa-gfs-original-v1'
DECODE_LOCK = threading.Lock()
FIELDS = {'temperature': ('TMP', '2 m above ground'),
          'u': ('UGRD', '100 m above ground'),
          'v': ('VGRD', '100 m above ground')}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path, value):
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())


def request(client, method, url, **kwargs):
    """Bounded retries only for transport failures, throttling and server errors."""
    for attempt in range(4):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
        except httpx.TransportError:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def publication_time(headers, issued, origin):
    """S3 Last-Modified evidence, fail closed for multipart/unknown ETags.

Multipart Last-Modified can be upload initiation, so it cannot establish completion.
For the public operational NOAA single-part objects we retain the provider's timestamp,
not an invented fixed publication lag or the present retrieval time.
"""
    headers = {k.lower(): v for k, v in headers.items()}
    etag = headers.get('etag', '').strip('"')
    if not re.fullmatch(r'[a-fA-F0-9]{32}', etag):
        raise ValueError('Cannot establish publication: multipart or unknown S3 ETag.')
    if headers.get('x-amz-server-side-encryption', 'AES256') != 'AES256':
        raise ValueError('Unsupported S3 encryption/ETag semantics.')
    try:
        available = instant(parsedate_to_datetime(headers['last-modified']))
    except (KeyError, TypeError, ValueError):
        raise ValueError('Missing or invalid provider Last-Modified publication evidence.') from None
    if not instant(issued) <= available <= instant(origin):
        raise ValueError(f'Object publication {available.isoformat()} outside [issued_at, origin].')
    return available


def parse_index(content, size, issued, lead):
    entries = [line.split(':') for line in content.decode('utf-8').splitlines() if line]
    if not entries or any(len(row) < 7 for row in entries):
        raise ValueError('Invalid NOAA GRIB index.')
    offsets = [int(row[1]) for row in entries]
    if offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])) or offsets[-1] >= size:
        raise ValueError('Invalid GRIB index offsets.')
    found = {}
    for i, row in enumerate(entries):
        for field, (parameter, level) in FIELDS.items():
            if row[3:5] != [parameter, level]:
                continue
            if field in found or row[2] != 'd=' + issued.strftime('%Y%m%d%H') or row[5] != f'{lead} hour fcst':
                raise ValueError('Duplicate field, wrong initialization, or non-forecast index entry.')
            found[field] = {'start': offsets[i], 'end': offsets[i + 1] - 1 if i + 1 < len(offsets) else size - 1,
                            'index_line': ':'.join(row)}
    if set(found) != set(FIELDS):
        raise ValueError('Missing 100 m wind components or 2 m temperature in GRIB.')
    return found


def checked_range(response, start, end, size, etag):
    if (response.status_code != 206
            or response.headers.get('content-range') != f'bytes {start}-{end}/{size}'
            or response.headers.get('etag') != etag
            or len(response.content) != end - start + 1):
        raise ValueError('GRIB range response/ETag mismatch; refusing partial or full-file response.')
    content = response.content
    if (content[:4] != b'GRIB' or content[7:8] != b'\x02' or content[-4:] != b'7777'
            or int.from_bytes(content[8:16], 'big') != len(content)):
        raise ValueError('Invalid or truncated original GRIB2 message.')
    return content


def interpolate(points, latitude, longitude):
    """Bilinear interpolation of scalar/vector components on regular lat/lon grid."""
    lats = sorted({float(p['lat']) for p in points})
    lons = sorted({float(p['lon']) for p in points})
    if (len(points) != 4 or len(lats) != 2 or len(lons) != 2
            or not lats[0] <= latitude <= lats[1] or not lons[0] <= longitude <= lons[1]):
        raise ValueError('GRIB neighbours do not bracket the turbine coordinates.')
    if len({(p['lat'], p['lon']) for p in points}) != 4:
        raise ValueError('Duplicate interpolation corner.')
    result, evidence = 0., []
    for point in points:
        lat, lon, value = float(point['lat']), float(point['lon']), float(point['value'])
        weight = ((latitude-lats[0]) if lat == lats[1] else (lats[1]-latitude)) / (lats[1]-lats[0])
        weight *= ((longitude-lons[0]) if lon == lons[1] else (lons[1]-longitude)) / (lons[1]-lons[0])
        if not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError('Missing/non-finite GRIB grid value.')
        result += weight * value
        evidence.append({'latitude': lat, 'longitude': lon, 'weight': weight, 'value': value})
    return result, evidence


def decode(content, field, issued, lead, locations):
    import eccodes as ec
    # ecCodes uses native shared state; serialize decoding while HTTP runs concurrently.
    with DECODE_LOCK:
        handle = ec.codes_new_from_message(content)
        try:
            expected_parameter = {'temperature': (0, 0, 2), 'u': (2, 2, 100), 'v': (2, 3, 100)}[field]
            identity = tuple(ec.codes_get(handle, key) for key in ['parameterCategory', 'parameterNumber', 'level'])
            reference = str(ec.codes_get(handle, 'dataDate')) + f"{ec.codes_get(handle, 'dataTime'):04d}"
            valid = str(ec.codes_get(handle, 'validityDate')) + f"{ec.codes_get(handle, 'validityTime'):04d}"
            expected_units = 'K' if field == 'temperature' else 'm s**-1'
            if (identity != expected_parameter or ec.codes_get(handle, 'discipline') != 0
                    or ec.codes_get(handle, 'typeOfLevel') != 'heightAboveGround'
                    or ec.codes_get(handle, 'units') != expected_units
                    or ec.codes_get(handle, 'stepType') != 'instant'
                    or ec.codes_get(handle, 'centre') != 'kwbc'
                    or ec.codes_get_long(handle, 'typeOfProcessedData') != 1
                    or ec.codes_get(handle, 'productionStatusOfProcessedData') != 0
                    or reference != issued.strftime('%Y%m%d%H%M')
                    or valid != (issued + pd.Timedelta(hours=lead)).strftime('%Y%m%d%H%M')
                    or ec.codes_get(handle, 'gridType') != 'regular_ll'
                    or ec.codes_get(handle, 'iDirectionIncrementInDegrees') != .25
                    or ec.codes_get(handle, 'jDirectionIncrementInDegrees') != .25):
                raise ValueError('GRIB metadata mismatch: need operational NOAA GFS forecast, correct units/level/time/grid.')
            values, evidence = {}, {}
            for turbine, (lat, lon) in locations.items():
                points = ec.codes_grib_find_nearest(handle, lat, lon, npoints=4)
                values[str(turbine)], evidence[str(turbine)] = interpolate(points, lat, lon)
            return values, {'units': expected_units, 'grid': 'regular_ll 0.25 degrees',
                            'level_m': expected_parameter[2], 'corners': evidence}
        finally:
            ec.codes_release(handle)


def cached_hour(directory, issued, lead, origin, locations):
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_locations = {str(k): list(v) for k, v in locations.items()}
    if (manifest['version'] != VERSION or manifest['locations'] != expected_locations
            or manifest['issued_at'] != issued.isoformat() or manifest['lead'] != lead):
        raise ValueError(f'Incompatible cache: {directory}. Choose a new --raw-dir.')
    if ({item['name'] for item in manifest['files']} != {'source.idx', 'temperature.grib2', 'u.grib2', 'v.grib2'}
            or len(manifest['files']) != 4):
        raise ValueError('Incomplete raw source cache.')
    for item in manifest['files']:
        path = directory / item['name']
        if (not path.exists() or path.stat().st_size != item['size']
                or sha256(path.read_bytes()) != item['sha256']):
            raise ValueError(f'Raw cache SHA-256 mismatch: {path}. Restore or choose a new --raw-dir.')
    available = max(publication_time(manifest['object_headers'], issued, origin),
                    publication_time(manifest['index_headers'], issued, origin))
    if available.isoformat() != manifest['available_at']:
        raise ValueError('Cached available_at disagrees with provider headers.')
    return manifest


def download_hour(client, raw_dir, issued, lead, origin, locations):
    directory = raw_dir / issued.strftime('%Y%m%dT%HZ') / f'f{lead:03d}'
    cached = cached_hour(directory, issued, lead, origin, locations)
    if cached:
        return cached
    key = f'gfs.{issued:%Y%m%d}/{issued:%H}/atmos/gfs.t{issued:%H}z.pgrb2.0p25.f{lead:03d}'
    url = BASE + '/' + key
    head = request(client, 'HEAD', url)
    index = request(client, 'GET', url + '.idx')
    # Preserve evidence even when timestamps fail validation; no final CSV is exported.
    write_json(directory / 'http-evidence.json', {
        'url': url, 'retrieved_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'object_headers': dict(head.headers), 'index_headers': dict(index.headers)})
    available = max(publication_time(head.headers, issued, origin),
                    publication_time(index.headers, issued, origin))
    size = int(head.headers['content-length'])
    ranges = parse_index(index.content, size, issued, lead)
    write_atomic(directory / 'source.idx', index.content)
    files = [{'name': 'source.idx', 'sha256': sha256(index.content), 'size': len(index.content)}]
    values, decoding = {}, {}
    for field, byte_range in ranges.items():
        start, end = byte_range['start'], byte_range['end']
        response = request(client, 'GET', url, headers={
            'Range': f'bytes={start}-{end}', 'If-Match': head.headers['etag']})
        content = checked_range(response, start, end, size, head.headers['etag'])
        if response.headers.get('last-modified') != head.headers['last-modified']:
            raise ValueError('Source changed between HEAD and range GET.')
        filename = field + '.grib2'
        write_atomic(directory / filename, content)
        files.append({'name': filename, 'sha256': sha256(content), 'size': len(content),
                      **byte_range, 'response_headers': dict(response.headers)})
        values[field], decoding[field] = decode(content, field, issued, lead, locations)
    manifest = {'version': VERSION, 'url': url, 'issued_at': issued.isoformat(), 'lead': lead,
                'valid_time': (issued + pd.Timedelta(hours=lead)).isoformat(),
                'available_at': available.isoformat(), 'publication_basis': 'public NOAA S3 single-part object and index Last-Modified',
                'retrieved_at': pd.Timestamp.now(tz='UTC').isoformat(),
                'object_headers': dict(head.headers), 'index_headers': dict(index.headers),
                'locations': {str(k): list(v) for k, v in locations.items()},
                'files': files, 'values': values, 'decoding': decoding}
    write_json(directory / 'manifest.json', manifest)
    return manifest


def schedule(first=FIRST_ORIGIN, last=LAST_ORIGIN):
    first, last = instant(first), instant(last)
    if first.hour != 18 or last.hour != 18 or first != first.floor('h') or last != last.floor('h'):
        raise ValueError('Agreed origins are daily 18:00 UTC (23:00 UTC+5).')
    if last < first or last - first > pd.Timedelta(days=365):
        raise ValueError('Invalid date interval (maximum 366 daily runs).')
    return pd.date_range(first, last, freq='24h')


def validate_export(path, origins):
    frame = validate_archive_frame(pd.read_csv(path))
    expected = len(origins) * 2 * 48
    if len(frame) != expected or set(frame.turbine) != {1, 2}:
        raise ValueError(f'Expected {expected} rows, two turbines, 48 hours per daily origin.')
    selections = []
    for origin in origins:
        for turbine in (1, 2):
            for horizon in (24, 48):
                selected, metadata = archive(path, turbine, origin, horizon)
                if instant(metadata['issued_at']) != origin.normalize() + pd.Timedelta(hours=12):
                    raise ValueError('Export selected an unexpected GFS cycle.')
                if horizon == 48:
                    selections.append({'turbine': turbine, 'origin': origin.isoformat(),
                                       'hours': len(selected), 'issued_at': metadata['issued_at'],
                                       'available_at': metadata['available_at']})
    return {'rows': len(frame), 'origins': len(origins), 'turbines': 2,
            'horizons_checked': [24, 48], 'duplicates': 0, 'missing_hours': 0,
            'csv_sha256': sha256(Path(path).read_bytes()), 'selections': selections}


def validate_bundle(output, origins, decode_raw=False):
    """Offline verification of CSV, run manifests, raw hashes and publication evidence."""
    output = Path(output)
    report = validate_export(output, origins)
    bundle = json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if bundle['version'] != VERSION or bundle['validation']['csv_sha256'] != report['csv_sha256']:
        raise ValueError('CSV SHA-256/version mismatch with provenance manifest.')
    if [item['origin'] for item in bundle['runs']] != [origin.isoformat() for origin in origins]:
        raise ValueError('Provenance does not cover the requested origins.')
    data = validate_archive_frame(pd.read_csv(output))
    raw_files, raw_bytes = 0, 0
    for run, origin in zip(bundle['runs'], origins):
        issued = origin.normalize() + pd.Timedelta(hours=12)
        if len(run['manifests']) != 48:
            raise ValueError('Provenance must contain 48 source-hour manifests per run.')
        hourly = []
        for lead, reference in zip(range(6, 54), run['manifests']):
            path = output.parent / reference['path']
            if sha256(path.read_bytes()) != reference['sha256']:
                raise ValueError(f'Source manifest SHA-256 mismatch: {path}')
            item = cached_hour(path.parent, issued, lead, origin, COORDINATES)
            if item is None:
                raise ValueError(f'Missing source-hour manifest: {path}')
            if {entry['name'] for entry in item['files']} != {'source.idx', 'temperature.grib2', 'u.grib2', 'v.grib2'}:
                raise ValueError('Missing required original source files.')
            if decode_raw:
                for field in FIELDS:
                    values, _ = decode((path.parent / (field + '.grib2')).read_bytes(),
                                       field, issued, lead, COORDINATES)
                    if any(abs(values[key] - item['values'][field][key]) > 1e-10 for key in values):
                        raise ValueError('Stored decoding differs from original GRIB messages.')
            hourly.append(item)
            raw_files += len(item['files'])
            raw_bytes += sum(entry['size'] for entry in item['files'])
        available = max(instant(item['available_at']) for item in hourly)
        run_data = data[data.issued_at == issued]
        if (instant(run['available_at']) != available or not run_data.available_at.eq(available).all()
                or not run_data.source.eq(run['source']).all()):
            raise ValueError('Exported publication/source disagrees with original provider evidence.')
        for item in hourly:
            for turbine in (1, 2):
                row = run_data[(run_data.turbine == turbine) & (run_data.valid_time == instant(item['valid_time']))]
                key = str(turbine)
                expected_wind = math.hypot(item['values']['u'][key], item['values']['v'][key])
                expected_temp = item['values']['temperature'][key] - 273.15
                if (len(row) != 1 or abs(row.iloc[0].wind_speed - expected_wind) > 1e-7
                        or abs(row.iloc[0].temperature - expected_temp) > 1e-7):
                    raise ValueError('CSV differs from source decoding/conversion.')
    report.update(raw_files_verified=raw_files, raw_bytes_verified=raw_bytes,
                  grib_messages_decoded=len(origins) * 48 * 3 if decode_raw else 0)
    return report


def fetch(output, raw_dir, first=FIRST_ORIGIN, last=LAST_ORIGIN, workers=6):
    output, raw_dir = Path(output), Path(raw_dir)
    origins = schedule(first, last)
    if not 1 <= workers <= 12:
        raise ValueError('workers must be between 1 and 12.')
    rows, runs = [], []
    with httpx.Client(timeout=90, limits=httpx.Limits(max_connections=workers)) as client:
        for origin in origins:
            issued = origin.normalize() + pd.Timedelta(hours=12)
            # f006..f053 inclusive: every hour in [origin, origin + 48h).
            with ThreadPoolExecutor(max_workers=workers) as executor:
                hourly = list(executor.map(lambda lead: download_hour(
                    client, raw_dir, issued, lead, origin, COORDINATES), range(6, 54)))
            available = max(instant(item['available_at']) for item in hourly)
            source = f'NOAA/GFS/0p25/{issued:%Y%m%dT%HZ}/wind100m/temp2m/bilinear'
            for item in hourly:
                for turbine in (1, 2):
                    key = str(turbine)
                    rows.append({'turbine': turbine, 'issued_at': issued.isoformat(),
                                 'available_at': available.isoformat(), 'valid_time': item['valid_time'],
                                 'wind_speed': math.hypot(item['values']['u'][key], item['values']['v'][key]),
                                 'temperature': item['values']['temperature'][key] - 273.15,
                                 'source': source, 'kind': 'forecast'})
            manifests = [raw_dir / issued.strftime('%Y%m%dT%HZ') / f'f{lead:03d}' / 'manifest.json' for lead in range(6, 54)]
            runs.append({'origin': origin.isoformat(), 'issued_at': issued.isoformat(),
                         'available_at': available.isoformat(), 'source': source,
                         'manifests': [{'path': Path(os.path.relpath(p, output.parent)).as_posix(),
                                        'sha256': sha256(p.read_bytes())} for p in manifests]})
            print(f'{origin.isoformat()}: 96 rows; published by {available.isoformat()}', flush=True)
    frame = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS).sort_values(['issued_at', 'turbine', 'valid_time'])
    # Validate a staging export before replacing an existing successful CSV.
    staging = output.with_name(output.name + '.staging')
    write_atomic(staging, frame.to_csv(index=False, float_format='%.8f').encode())
    report = validate_export(staging, origins)
    manifest = {'version': VERSION, 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
                'first_origin': origins[0].isoformat(), 'last_origin': origins[-1].isoformat(),
                'wind_units': 'm/s', 'wind_height_m': 100, 'temperature_units': 'degC',
                'temperature_height_m': 2, 'locations': COORDINATES,
                'interpolation': 'bilinear U/V then hypot; bilinear temperature; no time interpolation',
                'raw_storage': 'unaltered GRIB2 message byte ranges, complete source indexes and HTTP metadata',
                'runs': runs, 'validation': report}
    write_json(output.with_suffix('.manifest.json'), manifest)
    staging.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'validate'])
    parser.add_argument('--output', type=Path, default=ROOT / 'data/weather_forecasts.csv')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw/gfs')
    parser.add_argument('--first-origin', default=FIRST_ORIGIN)
    parser.add_argument('--last-origin', default=LAST_ORIGIN)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--decode-raw', action='store_true',
                        help='During validate, independently re-decode all original GRIB messages.')
    args = parser.parse_args()
    try:
        if args.command == 'fetch':
            result = fetch(args.output, args.raw_dir, args.first_origin, args.last_origin, args.workers)
        else:
            result = validate_bundle(args.output, schedule(args.first_origin, args.last_origin), args.decode_raw)
        print(json.dumps({k: v for k, v in result.items() if k != 'selections'}, indent=2))
    except (ValueError, OSError, httpx.HTTPError, ImportError) as error:
        parser.exit(2, f'Weather archive failed; no successful export claimed: {error}\n')


if __name__ == '__main__':
    main()
