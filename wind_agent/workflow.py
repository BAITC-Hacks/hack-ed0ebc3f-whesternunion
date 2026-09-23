import hashlib
import importlib.util
import json
import platform
import uuid
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import service, weather_download
from .config import ARTIFACT_DIR, ROOT, TIMEZONE
from .data import instant, load_history, training_end
from .schedule import FIRST_ORIGIN, LAST_ORIGIN, SCORE_START, SCORE_END


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


CODE_HASHES = {path.name: file_hash(path) for path in sorted((ROOT / 'wind_agent').glob('*.py'))}


def check_replay(result, origins):
    rows = result['predictions']
    if len(rows) != len(origins) * result['horizon']:
        raise ValueError('Replay does not contain every issued forecast hour.')
    expected = {(origin, target) for origin in origins
                for target in pd.date_range(origin, periods=result['horizon'], freq='h')}
    actual = {(instant(row['origin']), instant(row['timestamp'])) for row in rows}
    if actual != expected or result['coverage']['missing_hours']:
        raise ValueError('Replay has missing or unexpected forecast hours.')
    for row in rows:
        origin = instant(row['origin'])
        if not instant(row['weather_issued_at']) <= instant(row['weather_available_at']) <= origin:
            raise ValueError('Replay used weather unavailable at issue time.')
        if instant(row['training_cutoff']) > min(origin, training_end()):
            raise ValueError('Replay trained on future observations.')
        values = [row['lower'], row['power'], row['upper']]
        if not np.isfinite(values).all() or not 0 <= values[0] <= values[1] <= values[2] <= 1:
            raise ValueError('Replay contains invalid power or interval bounds.')


def outputs_unchanged(report):
    outputs = report.get('output_sha256', {})
    if not isinstance(outputs, dict):
        return False
    try:
        return bool(outputs) and all(Path(path).is_file() and file_hash(path) == digest
                                     for path, digest in outputs.items())
    except OSError:
        return False


class CaseAgent:
    def __init__(self, directory, offline=False, workers=6):
        self.directory = Path(directory)
        self.offline = offline
        self.workers = workers
        self.report = {'id': uuid.uuid4().hex, 'status': 'running', 'trace': []}
        self.journal = self.directory / f"case-{self.report['id']}.json"

    def record(self, action, reason, **details):
        self.report['trace'].append({'at': pd.Timestamp.now(tz='UTC').isoformat(),
                                     'action': action, 'reason': reason, **details})
        weather_download.write_json(self.journal, self.report)
        print(f'[{action}] {reason}', flush=True)

    def ensure_weather(self, path, raw_dir, origins):
        path = Path(path)
        try:
            validation = weather_download.validate_bundle(path, origins)
        except FileNotFoundError:
            if self.offline:
                raise ValueError(f'Offline mode requires the complete weather bundle: {path}') from None
            if importlib.util.find_spec('eccodes') is None:
                raise ValueError('Install requirements-weather.txt to download original GRIB weather.') from None
            self.record('fetch_weather', 'Missing weather or publication evidence; download/resume original NOAA files.',
                        path=str(path), first_origin=origins[0].isoformat(), last_origin=origins[-1].isoformat())
            weather_download.fetch(path, raw_dir, origins[0], origins[-1], self.workers)
            validation = weather_download.validate_bundle(path, origins)
        self.record('verify_weather', 'Publication evidence, raw hashes and forecast coverage verified.',
                    path=str(path), sha256=validation['csv_sha256'], rows=validation['rows'])
        return validation

    def replay(self, path, origins, score_start, score_end, training_path, phase):
        reports = []
        for turbine in (1, 2):
            for horizon in (24, 48):
                self.record('forecast', 'Run chronological forecasts and select the model using only prior data.',
                            phase=phase, turbine=turbine, horizon=horizon)
                result = service.backtest(turbine, path, origins[0], origins[-1] + pd.Timedelta(days=1),
                                          horizon, score_start=score_start, score_end=score_end,
                                          training_archive_path=training_path)
                check_replay(result, origins)
                summary = {key: value for key, value in result.items() if key != 'predictions'}
                summary['models'] = sorted({row['model'] for row in result['predictions']})
                reports.append(summary)
                self.record('verify_forecast', 'Every hour present; power bounds and as-of constraints satisfied.',
                            phase=phase, turbine=turbine, horizon=horizon,
                            models=summary['models'], scored_predictions=result['scored_predictions'])
        return reports


def write_case_summary(report, path):
    lines = ['# WindPilot: проверенный запуск', '',
             f"Окно оценки: {report['score_start']} — {report['score_end']} (конец исключён).",
             '', '## Прогнозы тестового периода', '',
             '| Турбина | Горизонт | Строк | Пропусков часов | Фактов для оценки | MAE |',
             '|---|---:|---:|---:|---:|---:|']
    for item in report['reports']:
        mae = f"{item['metrics']['mae']:.6f}" if item['metrics'] else 'нет факта'
        lines.append(f"| {item['turbine']} | {item['horizon']} | {item['total_predictions']} | "
                     f"{item['coverage']['missing_hours']} | {item['scored_predictions']} | {mae} |")
    lines.extend(['', '## Проверка до тестового периода', '',
                  '| Турбина | Горизонт | Пар для сравнения | MAE модели | MAE инерции |',
                  '|---|---:|---:|---:|---:|'])
    for item in report['validation_reports']:
        comparison = item['baseline_comparison']
        if comparison:
            lines.append(f"| {item['turbine']} | {item['horizon']} | {comparison['paired_predictions']} | "
                         f"{comparison['model']['mae']:.6f} | {comparison['power_persistence_6h']['mae']:.6f} |")
    if not report['validation_reports']:
        lines.extend(['', 'Предтестовая проверка отключена: для неё требуется не менее 14 суток погодной истории.'])
    lines.extend(['', 'Эта проверка относится только к периоду до первого тестового выпуска.',
                  'Февральские ошибки доступны только при наличии февральских измерений.',
                  'Мощность нормализована; интервалы диагностические. Часовой пояс SCADA требует подтверждения.',
                  '', '## Происхождение', '', f"Fingerprint: `{report['input_fingerprint']}`",
                  f"Погода: `{report['weather_sha256']}`",
                  f"Погода для обучения: `{report['training_weather_sha256']}`", ''])
    weather_download.write_atomic(path, '\n'.join(lines).encode('utf-8'))


def run_case(archive_path=None, raw_dir=None, first_origin=FIRST_ORIGIN, last_origin=LAST_ORIGIN,
             score_start=SCORE_START, score_end=SCORE_END, training_days=14,
             training_archive=None, workers=6, offline=False):
    archive_path = Path(archive_path or ROOT / 'data/weather_forecasts.csv').resolve()
    raw_dir = Path(raw_dir or ROOT / 'data/raw/gfs').resolve()
    training_path = Path(training_archive or archive_path.with_name('weather_training.csv')).resolve()
    if not 0 <= training_days <= 60:
        raise ValueError('training_days must be between 0 and 60.')
    if not 1 <= workers <= 12:
        raise ValueError('workers must be between 1 and 12.')
    if training_days and training_path == archive_path:
        raise ValueError('Training and test weather archives must use separate paths.')
    origins = weather_download.schedule(first_origin, last_origin)
    score_start, score_end = instant(score_start), instant(score_end)
    if (score_start != score_start.floor('h') or score_end != score_end.floor('h')
            or not origins[0] <= score_start < score_end <= origins[-1] + pd.Timedelta(days=1)):
        raise ValueError('Scoring window must be hourly and covered by all 24-hour releases.')
    agent = CaseAgent(ARTIFACT_DIR, offline, workers)
    latest_path = agent.directory / 'case-latest.json'
    try:
        quality = {str(turbine): load_history(turbine, min(origins[0], training_end()))[1]
                   for turbine in (1, 2)}
        agent.record('audit', 'Both SCADA datasets are readable; incomplete hours excluded.')
        weather_validation = agent.ensure_weather(archive_path, raw_dir, origins)
        training_validation = None
        training_origins = []
        if training_days:
            training_origins = weather_download.schedule(origins[0] - pd.Timedelta(days=training_days),
                                                          origins[0] - pd.Timedelta(days=1))
            training_validation = agent.ensure_weather(training_path, raw_dir, training_origins)
        environment = {'python': platform.python_version(),
                       **{name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')}}
        inputs = {'code': CODE_HASHES, 'environment': environment, 'timezone': TIMEZONE,
                  'scada': {key: value['sha256'] for key, value in quality.items()},
                  'weather': weather_validation['csv_sha256'],
                  'training_weather': training_validation['csv_sha256'] if training_validation else None,
                  'origins': [origin.isoformat() for origin in origins],
                  'score_start': score_start.isoformat(), 'score_end': score_end.isoformat()}
        fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        if latest_path.exists():
            try:
                previous = json.loads(latest_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                previous = None
            if (isinstance(previous, dict) and previous.get('status') == 'complete'
                    and previous.get('input_fingerprint') == fingerprint and outputs_unchanged(previous)):
                agent.report.update(status='unchanged', input_fingerprint=fingerprint,
                                    previous_report=str(latest_path))
                agent.record('reuse', 'Verified inputs and output files unchanged; no repeated training or replay.')
                return {**previous, 'cached': True}
        agent.record('recalculate', 'Inputs or results changed; a new verified replay is required.')
        validation_reports = []
        if training_days >= 14:
            validation_reports = agent.replay(training_path, training_origins[-3:], training_origins[-3],
                                              origins[0], training_path, 'pretest_validation')
        reports = agent.replay(archive_path, origins, score_start, score_end,
                               training_path if training_days else None, 'test')
        agent.report.update(status='complete', cached=False, input_fingerprint=fingerprint,
                            inputs=inputs, quality=quality, score_start=score_start.isoformat(),
                            score_end=score_end.isoformat(), reports=reports,
                            validation_reports=validation_reports,
                            weather_sha256=inputs['weather'], training_weather_sha256=inputs['training_weather'],
                            evaluation_status='scored' if any(item['metrics'] for item in reports) else 'no_actuals')
        summary_path = agent.directory / f"case-{agent.report['id']}.md"
        write_case_summary(agent.report, summary_path)
        output_paths = [summary_path]
        for item in reports + validation_reports:
            output_paths.extend(Path(path) for path in item['artifacts'].values())
            output_paths.extend(service.ARTIFACT_DIR / 'runs' / f'{run_id}.json' for run_id in item['runs'])
        agent.report['output_sha256'] = {str(path.resolve()): file_hash(path) for path in output_paths}
        agent.report['summary_path'] = str(summary_path.resolve())
        agent.record('complete', 'All forecasts and provenance checks passed; report saved.',
                     evaluation_status=agent.report['evaluation_status'])
        weather_download.write_json(latest_path, agent.report)
        return agent.report
    except Exception as error:
        agent.report['status'] = 'failed'
        agent.record('stop', 'Case not completed; inspect the error and retained download cache.',
                     error=str(error), error_type=type(error).__name__)
        raise
