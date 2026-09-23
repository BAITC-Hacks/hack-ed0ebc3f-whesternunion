import csv
import io
import json
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ARTIFACT_DIR, ROOT, TIMEZONE
from .data import load_history
from .service import run_forecast

app = FastAPI(title='WindPilot · HackAlem', version='0.1.0')
app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')


class ForecastRequest(BaseModel):
    turbine: Literal[1, 2] = 1
    horizon: Literal[24, 48] = 48
    mode: Literal['baseline', 'archive', 'live'] = 'baseline'
    origin: str | None = None
    use_ai: bool = False


@app.get('/')
def index():
    return FileResponse(ROOT / 'web' / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'timezone': TIMEZONE}


@app.get('/api/datasets')
def datasets():
    result = []
    for turbine in (1, 2):
        try:
            _, quality = load_history(turbine)
            result.append({'turbine': turbine, 'quality': quality})
        except ValueError as error:
            result.append({'turbine': turbine, 'error': str(error)})
    return result


@app.post('/api/forecast')
def forecast(request: ForecastRequest):
    try:
        return run_forecast(**request.model_dump(), archive_path=ROOT / 'data' / 'weather_forecasts.csv')
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(422, str(error)) from None
    except (httpx.HTTPError, KeyError):
        raise HTTPException(502, 'Не удалось получить погодный прогноз. Проверьте сеть и настройки источника.') from None


def stored_run(run_id):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise HTTPException(404, 'Запуск не найден.')
    path = ARTIFACT_DIR / 'runs' / f'{run_id}.json'
    if not path.exists():
        raise HTTPException(404, 'Запуск не найден.')
    return json.loads(path.read_text(encoding='utf-8'))


@app.get('/api/runs')
def runs():
    paths = sorted((ARTIFACT_DIR / 'runs').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    return [{key: result[key] for key in ['id', 'created_at', 'turbine', 'mode', 'horizon', 'origin']}
            for result in (json.loads(path.read_text(encoding='utf-8')) for path in paths)]


@app.get('/api/runs/{run_id}')
def get_run(run_id: str):
    return stored_run(run_id)


@app.get('/api/runs/{run_id}/csv')
def export_csv(run_id: str):
    result = stored_run(run_id)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=['turbine', 'origin', *result['forecast'][0].keys()])
    writer.writeheader()
    writer.writerows({'turbine': result['turbine'], 'origin': result['origin'], **row} for row in result['forecast'])
    return Response(buffer.getvalue(), media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="forecast-{run_id}.csv"'})
