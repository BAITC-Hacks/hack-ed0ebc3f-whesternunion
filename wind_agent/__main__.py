import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from .data import load_history
from .agent import explain
from .service import backtest, run_forecast, save_result


def watch_forecasts(args):
    previous = None
    failures = 0
    while True:
        try:
            result = run_forecast(args.turbine, args.horizon, args.origin,
                                  args.mode, args.archive, persist=False)
            changed = result['input_fingerprint'] != previous
            if changed:
                if args.ai:
                    result['ai'] = explain(result)
                save_result(result)
            print(json.dumps({'id': result['id'], 'changed': changed,
                              'summary': result['summary'], 'ai': result['ai']},
                             ensure_ascii=False, indent=2), flush=True)
            previous = result['input_fingerprint']
            failures = 0
            delay = args.interval
        except (ValueError, FileNotFoundError, OSError, httpx.HTTPError) as error:
            failures += 1
            delay = min(args.interval, 60 * 2 ** min(failures - 1, 6))
            print(f'Ошибка цикла: {error}. Повтор через {delay} с.', file=sys.stderr, flush=True)
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description='WindPilot: local forecasting tools')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--turbine', type=int, choices=[1, 2], default=1)
    for name in ['forecast', 'watch']:
        command = commands.add_parser(name)
        command.add_argument('--turbine', type=int, choices=[1, 2], default=1)
        command.add_argument('--horizon', type=int, choices=[24, 48], default=48)
        command.add_argument('--mode', choices=['baseline', 'archive', 'live'],
                             default='live' if name == 'watch' else 'baseline')
        command.add_argument('--origin')
        command.add_argument('--archive', type=Path)
        command.add_argument('--ai', action='store_true')
        if name == 'watch':
            command.add_argument('--interval', type=int, default=3600)
    replay = commands.add_parser('backtest')
    replay.add_argument('--turbine', choices=['1', '2', 'all'], default='all')
    replay.add_argument('--archive', type=Path, required=True)
    replay.add_argument('--start', default='2026-01-31T00:00:00+05:00')
    replay.add_argument('--end', default='2026-03-01T00:00:00+05:00')
    replay.add_argument('--score-start')
    replay.add_argument('--horizon', choices=['24', '48', 'both'], default='both')
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            print(json.dumps(load_history(args.turbine)[1], ensure_ascii=False, indent=2))
        elif args.command == 'backtest':
            turbines = (1, 2) if args.turbine == 'all' else (int(args.turbine),)
            horizons = (24, 48) if args.horizon == 'both' else (int(args.horizon),)
            score_start = args.score_start or (
                '2026-02-01T00:00:00+05:00' if args.start == '2026-01-31T00:00:00+05:00'
                else args.start)
            reports = []
            for turbine in turbines:
                for horizon in horizons:
                    result = backtest(turbine, args.archive, args.start, args.end,
                                      horizon, score_start=score_start)
                    reports.append({key: value for key, value in result.items()
                                    if key != 'predictions'})
            print(json.dumps(reports[0] if len(reports) == 1 else {'reports': reports},
                             ensure_ascii=False, indent=2))
        else:
            if args.command == 'watch' and args.interval < 60:
                parser.error('--interval должен быть не менее 60 секунд.')
            if args.mode == 'archive' and args.archive is None:
                parser.error('Для архивного режима нужен --archive.')
            if args.command == 'watch':
                watch_forecasts(args)
            else:
                result = run_forecast(args.turbine, args.horizon, args.origin,
                                      args.mode, args.archive, args.ai)
                print(json.dumps({'id': result['id'], 'summary': result['summary'],
                                  'ai': result['ai']}, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Ошибка: {error}\n')
    except httpx.HTTPError:
        parser.exit(2, 'Не удалось получить прогноз погоды. Проверьте доступ к Open-Meteo.\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
