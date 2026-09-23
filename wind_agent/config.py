import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
DATA_DIR = Path(os.getenv('DATA_DIR', str(ROOT)))
ARTIFACT_DIR = Path(os.getenv('ARTIFACT_DIR', str(ROOT / 'artifacts')))
TIMEZONE = os.getenv('DATA_TIMEZONE', 'Etc/GMT-5')
TRAIN_END = '2026-02-01T00:00:00'
COORDINATES = {1: (43.645150, 78.535604), 2: (43.643198, 78.538828)}


def coordinates(turbine: int) -> tuple[float, float]:
    try:
        lat = float(os.getenv(f'TURBINE_{turbine}_LAT') or COORDINATES[turbine][0])
        lon = float(os.getenv(f'TURBINE_{turbine}_LON') or COORDINATES[turbine][1])
    except (KeyError, ValueError):
        raise ValueError('Укажите координаты турбины в .env.') from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Некорректные координаты турбины.')
    return lat, lon
