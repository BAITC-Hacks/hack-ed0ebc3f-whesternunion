import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from .data import instant


def features(frame: pd.DataFrame) -> pd.DataFrame:
    # Only weather available in the forecast, never future power or measured wind.
    return pd.DataFrame({
        'wind': frame.wind_speed,
        'wind_cubed': frame.wind_speed ** 3,
        'temperature': frame.temperature,
    }, index=frame.index)


def forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = features(frame)
    result['lead_hours'] = frame.lead_hours
    return result


def source_family(source: str) -> str:
    parts = str(source).split('/')
    return '/'.join(parts[:2]) if len(parts) >= 3 else str(source)


def archived_training_samples(history, archive_path, turbine, cutoff, source):
    archive = pd.read_csv(archive_path)
    required = {'turbine', 'issued_at', 'available_at', 'valid_time', 'wind_speed',
                'temperature', 'source', 'kind'}
    if not required.issubset(archive.columns):
        raise ValueError('Архив погоды не содержит поля для обучения на прогнозах.')
    archive = archive[(archive.turbine == turbine) & (archive.kind == 'forecast')].copy()
    archive = archive[archive.source.map(source_family) == source_family(source)].copy()
    if archive.empty:
        return archive
    for column in ['issued_at', 'available_at', 'valid_time']:
        archive[column] = archive[column].map(instant)
    if archive.duplicated(['source', 'issued_at', 'valid_time']).any():
        raise ValueError('Архив погоды содержит дубликаты прогнозных часов для обучения.')
    lead = (archive.valid_time - archive.issued_at) / pd.Timedelta(hours=1)
    eligible = ((archive.issued_at <= archive.available_at)
                & (archive.available_at <= archive.valid_time)
                & (archive.valid_time + pd.Timedelta(hours=1) <= instant(cutoff))
                & lead.between(0, 72))
    archive = archive.loc[eligible].copy()
    archive['lead_hours'] = lead.loc[eligible].astype(float)
    archive['wind_speed'] = pd.to_numeric(archive.wind_speed, errors='coerce')
    archive['temperature'] = pd.to_numeric(archive.temperature, errors='coerce')
    archive['power'] = archive.valid_time.map(history.power)
    archive = archive[(archive.wind_speed.between(0, 75))
                      & (archive.temperature.between(-80, 65))
                      & np.isfinite(archive[['wind_speed', 'temperature', 'power']]).all(axis=1)]
    return archive.set_index('valid_time').sort_index()


@threadpool_limits.wrap(limits=1)
def fit_archive_model(history, archive_path, turbine, cutoff, source):
    samples = archived_training_samples(history, archive_path, turbine, cutoff, source)
    unique_hours = samples.index.unique()
    if (len(unique_hours) < 240 or samples.lead_hours.nunique() < 12
            or samples.lead_hours.min() > 12 or samples.lead_hours.max() < 36):
        return None
    boundary = unique_hours[int(len(unique_hours) * .8)]
    train = samples.loc[samples.index < boundary]
    validation = samples.loc[samples.index >= boundary]
    def new_model():
        return HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15,
                                            l2_regularization=2, random_state=42,
                                            early_stopping=False)
    evaluation = new_model().fit(forecast_features(train), train.power)
    prediction = np.clip(evaluation.predict(forecast_features(validation)), 0, 1)
    residual = float(np.quantile(np.abs(validation.power.to_numpy() - prediction), .9))
    metrics = {
        'mae': float(mean_absolute_error(validation.power, prediction)),
        'rmse': float(np.sqrt(mean_squared_error(validation.power, prediction))),
        'constant_baseline_mae': float(mean_absolute_error(
            validation.power, np.full(len(validation), train.power.mean()))),
        'train_hours': int(train.index.nunique()),
        'validation_hours': int(validation.index.nunique()),
        'train_end': train.index[-1].isoformat(),
        'validation_start': validation.index[0].isoformat(),
        'validation_end': validation.index[-1].isoformat(),
        'scope': 'power_on_archived_forecast_weather',
        'source_family': source_family(source),
        'lead_min': float(samples.lead_hours.min()),
        'lead_max': float(samples.lead_hours.max()),
        'note': 'Хронологическая проверка на архивной прогнозной погоде до момента выпуска; это не оценка февраля.',
    }
    model = new_model().fit(forecast_features(samples), samples.power)
    return model, residual, metrics


@threadpool_limits.wrap(limits=1)
def fit_model(history: pd.DataFrame):
    clean = history.dropna().sort_index()
    if len(clean) < 240:
        raise ValueError('Для обучения нужно не менее 240 полных часов.')
    split = int(len(clean) * .8)
    train, validation = clean.iloc[:split], clean.iloc[split:]
    def new_model():
        return HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15,
                                            l2_regularization=2, random_state=42,
                                            early_stopping=False)
    evaluation = new_model().fit(features(train), train.power)
    pred = np.clip(evaluation.predict(features(validation)), 0, 1)
    residual = float(np.quantile(np.abs(validation.power.to_numpy() - pred), .9))
    metrics = {
        'mae': float(mean_absolute_error(validation.power, pred)),
        'rmse': float(np.sqrt(mean_squared_error(validation.power, pred))),
        'constant_baseline_mae': float(mean_absolute_error(
            validation.power, np.full(len(validation), train.power.mean()))),
        'train_hours': len(train), 'validation_hours': len(validation),
        'train_end': train.index[-1].isoformat(),
        'validation_start': validation.index[0].isoformat(),
        'validation_end': validation.index[-1].isoformat(),
        'scope': 'power_curve_on_observed_weather',
        'note': 'Проверка преобразования ветер → мощность на фактической погоде; это НЕ ошибка прогноза на 24–48 часов.',
    }
    model = new_model().fit(features(clean), clean.power)
    return model, residual, metrics


@threadpool_limits.wrap(limits=1)
def predict(model, weather: pd.DataFrame, residual: float, issued_at=None) -> pd.DataFrame:
    result = weather.copy()
    if issued_at is None:
        inputs = features(weather)
    else:
        forecast = weather.copy()
        forecast['lead_hours'] = (forecast.index - instant(issued_at)) / pd.Timedelta(hours=1)
        inputs = forecast_features(forecast)
    result['power'] = np.clip(model.predict(inputs), 0, 1)
    result['lower'] = np.clip(result.power - residual, 0, 1)
    result['upper'] = np.clip(result.power + residual, 0, 1)
    return result
