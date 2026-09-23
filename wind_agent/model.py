import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits


def features(frame: pd.DataFrame) -> pd.DataFrame:
    # Only weather available in the forecast, never future power or measured wind.
    return pd.DataFrame({
        'wind': frame.wind_speed,
        'wind_cubed': frame.wind_speed ** 3,
        'temperature': frame.temperature,
    }, index=frame.index)


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
def predict(model, weather: pd.DataFrame, residual: float) -> pd.DataFrame:
    result = weather.copy()
    result['power'] = np.clip(model.predict(features(weather)), 0, 1)
    result['lower'] = np.clip(result.power - residual, 0, 1)
    result['upper'] = np.clip(result.power + residual, 0, 1)
    return result
