from functools import lru_cache
from pathlib import Path
import hashlib
import json
import threading
import uuid

import numpy as np
import pandas as pd

from . import weather
from .agent import explain
from .config import ARTIFACT_DIR, TIMEZONE
from .data import dataset_path, instant, load_history, training_end
from .model import fit_archive_model, fit_model, predict

LOCK = threading.RLock()
MODEL_VERSION = 'power-curve-hgb-v1'
ARCHIVE_MODEL_VERSION = 'forecast-weather-hgb-v1'


@lru_cache(maxsize=8)
def trained(turbine, cutoff, source_hash):
    history, quality = load_history(turbine, cutoff)
    model, residual, validation = fit_model(history)
    return history, quality, model, residual, validation


@lru_cache(maxsize=8)
def archive_trained(turbine, cutoff, source_hash, archive_path, archive_hash, source):
    history = trained(turbine, cutoff, source_hash)[0]
    return fit_archive_model(history, archive_path, turbine, cutoff, source)


def save_result(result):
    directory = ARTIFACT_DIR / 'runs'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{result['id']}.json"
    temporary = directory / f'{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def run_forecast(turbine=1, horizon=48, origin=None, mode='baseline',
                 archive_path=None, use_ai=False, persist=True):
    with LOCK:
        trace = []
        def event(step, message):
            trace.append({'step': step, 'message': message,
                          'at': pd.Timestamp.now(tz='UTC').isoformat()})
        if mode not in {'baseline', 'archive', 'live'}:
            raise ValueError('Неизвестный режим прогноза.')
        if mode == 'live':
            forecast_weather, provenance, origin = weather.live(turbine, horizon)
        else:
            origin = instant(origin) if origin is not None else training_end()
        weather.target_hours(origin, horizon)
        cutoff = min(origin, training_end())
        source_hash = hashlib.sha256(dataset_path(turbine).read_bytes()).hexdigest()
        history, quality, model, residual, validation = trained(turbine, cutoff.isoformat(), source_hash)
        event('prepare', f"Проверено {quality['usable_hours']} часовых наблюдений; обучение до {cutoff.isoformat()}.")
        if mode == 'baseline':
            forecast_weather, provenance = weather.persistence(history, origin, horizon)
        elif mode == 'archive':
            if archive_path is None:
                raise ValueError('Для архивного режима нужен CSV прогнозов погоды.')
            forecast_weather, provenance = weather.archive(Path(archive_path), turbine, origin, horizon)
        event('weather', f"Получена погода: {provenance['source']}; {horizon} часов.")
        model_version = MODEL_VERSION
        issued_at = None
        if mode == 'archive':
            archived = archive_trained(turbine, cutoff.isoformat(), source_hash,
                                       str(Path(archive_path).resolve()), provenance['sha256'],
                                       provenance['source'])
            if archived is not None:
                model, residual, validation = archived
                model_version = ARCHIVE_MODEL_VERSION
                issued_at = provenance['issued_at']
        event('train', f"{model_version}: хронологическая валидация, затем обучение на доступной истории.")
        output = predict(model, forecast_weather, residual, issued_at)
        rows = []
        for timestamp, row in output.iterrows():
            rows.append({'timestamp': timestamp.isoformat(),
                         **{column: round(float(value), 6) for column, value in row.items()}})
        warnings = [
            f'Часовой пояс исходных CSV принят как {TIMEZONE}; требуется подтверждение.',
            'Нормализованная мощность 0–1. Для перевода в МВт и МВт·ч нужна номинальная мощность турбины.',
        ]
        if model_version == MODEL_VERSION:
            warnings.append('Диапазон основан на ошибках модели мощности при измеренной погоде; погодная неопределённость не учтена. Это не калиброванный интервал будущего прогноза.')
            if mode == 'archive':
                warnings.append('Недостаточно парных часов или покрытия заблаговременности для этой модели погоды до момента выпуска; используется модель на измеренном ветре.')
        else:
            warnings.append('Диапазон основан на остатках исторических прогнозов погоды; это диагностическая полоса, а не калиброванный интервал.')
        if mode == 'baseline':
            warnings.append('Baseline без внешнего прогноза погоды не удовлетворяет погодной части задания.')
        if mode == 'live':
            warnings.append('Высота измерения ветра в CSV неизвестна; прогноз ветра на 100 м пока не откалиброван.')
        if quality['missing_hours']:
            warnings.append(f"Часов с пропусками или недостаточным покрытием: {quality['missing_hours']}; они исключены из обучения.")
        ramps = int((output.power.diff().abs() > .25).sum())
        summary = {
            'mean_power': round(float(output.power.mean()), 4),
            'peak_power': round(float(output.power.max()), 4),
            'equivalent_full_load_hours': round(float(output.power.sum()), 3),
            'mean_wind_speed': round(float(output.wind_speed.mean()), 2),
            'large_ramps': ramps, 'hours': len(rows),
        }
        event('analyze', f'Проверены физические границы 0–1; резких изменений: {ramps}.')
        fingerprint = hashlib.sha256(json.dumps({
            'version': model_version, 'turbine': turbine, 'origin': origin.isoformat(),
            'mode': mode, 'horizon': horizon, 'source': source_hash,
            'weather': forecast_weather.to_json(date_format='iso'),
            'provenance': {k: v for k, v in provenance.items() if k != 'retrieved_at'}, 'timezone': TIMEZONE,
        }, sort_keys=True).encode()).hexdigest()
        result = {
            'id': uuid.uuid4().hex, 'input_fingerprint': fingerprint,
            'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'turbine': turbine, 'origin': origin.isoformat(), 'horizon': horizon,
            'mode': mode, 'model': model_version, 'training_cutoff': cutoff.isoformat(),
            'quality': quality, 'validation': validation, 'weather': provenance,
            'summary': summary, 'forecast': rows, 'warnings': warnings, 'trace': trace,
        }
        result['ai'] = explain(result) if use_ai else {'status': 'not_requested', 'text': 'AI-анализ не запрошен.'}
        if use_ai:
            event('ai', 'OpenAI-анализ: ' + result['ai']['status'])
        if persist:
            event('save', 'Прогноз и происхождение данных сохранены локально.')
            save_result(result)
        return result


def error_metrics(rows, prediction='power'):
    if not rows:
        return None
    error = np.array([row[prediction] - row['actual'] for row in rows])
    return {'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt((error ** 2).mean()))}


def recent_power_baseline(observations, origin):
    recent = observations.loc[(observations.index >= origin - pd.Timedelta(hours=6))
                              & (observations.index + pd.Timedelta(hours=1) <= origin), 'power'].dropna()
    if len(recent) < 4:
        return None
    return float(recent.mean())


def backtest(turbine, archive_path, start, end, horizon=24, persist=True, score_start=None):
    start, end = instant(start), instant(end)
    if end <= start or end - start > pd.Timedelta(days=62):
        raise ValueError('Период бэктеста должен быть от 1 до 62 дней.')
    score_start = instant(score_start) if score_start is not None else start
    if not start <= score_start < end:
        raise ValueError('Начало оценки должно попадать в период запусков.')
    observations, _ = load_history(turbine)
    rows, runs = [], []
    for origin in pd.date_range(start, end, freq='24h', inclusive='left'):
        result = run_forecast(turbine, horizon, origin, 'archive', archive_path, persist=persist)
        runs.append(result['id'])
        baseline_power = recent_power_baseline(observations, origin)
        for row in result['forecast']:
            timestamp = instant(row['timestamp'])
            actual = observations.power.get(timestamp, np.nan) if score_start <= timestamp < end else np.nan
            rows.append({**row, 'origin': origin.isoformat(),
                          'lead_hour': int((timestamp - origin).total_seconds() / 3600),
                          'actual': float(actual) if pd.notna(actual) else None,
                          'baseline_power': baseline_power})
    scored = [row for row in rows if row['actual'] is not None]
    comparable = [row for row in scored if row['baseline_power'] is not None]
    comparison = None
    if comparable:
        comparison = {'paired_predictions': len(comparable),
                      'model': error_metrics(comparable),
                      'power_persistence_6h': error_metrics(comparable, 'baseline_power')}
    by_lead = {}
    for lead in range(horizon):
        lead_rows = [row for row in scored if row['lead_hour'] == lead]
        by_lead[str(lead)] = {'scored_predictions': len(lead_rows),
                              'metrics': error_metrics(lead_rows)}
    result = {'turbine': turbine, 'start': start.isoformat(), 'end': end.isoformat(),
              'score_start': score_start.isoformat(), 'score_end': end.isoformat(),
              'horizon': horizon, 'runs': runs, 'predictions': rows,
              'scored_predictions': len(scored), 'total_predictions': len(rows),
              'metrics': error_metrics(scored), 'metrics_by_lead': by_lead,
              'baseline_comparison': comparison,
              'note': 'Все горизонты сохраняются. Метрики считаются только внутри окна оценки; перекрытия оцениваются отдельно по каждому запуску. Нет факта — нет метрики.'}
    if persist:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        name = f'backtest-t{turbine}-{uuid.uuid4().hex[:8]}'
        json_path = ARTIFACT_DIR / f'{name}.json'
        csv_path = ARTIFACT_DIR / f'{name}.csv'
        result['artifacts'] = {'json': str(json_path), 'csv': str(csv_path)}
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return result
