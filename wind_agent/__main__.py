import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from .data import load_history
from .agent import explain
from .service import backtest, run_forecast, save_result
from .schedule import FIRST_ORIGIN, ISSUE_END, LAST_ORIGIN, SCORE_END, SCORE_START
from .workflow import run_case


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


def case_command(args):
    failures = 0
    while True:
        try:
            result = run_case(archive_path=args.archive, raw_dir=args.raw_dir,
                              first_origin=args.first_origin, last_origin=args.last_origin,
                              score_start=args.score_start, score_end=args.score_end,
                              training_days=args.training_days,
                              training_archive=args.training_archive,
                              workers=args.workers, offline=args.offline)
            print(json.dumps({key: result[key] for key in (
                'id', 'status', 'cached', 'evaluation_status', 'summary_path')},
                ensure_ascii=False, indent=2), flush=True)
            failures = 0
            delay = args.interval
        except (ValueError, OSError, httpx.HTTPError) as error:
            if not args.watch:
                raise
            failures += 1
            delay = min(args.interval, 60 * 2 ** min(failures - 1, 6))
            print(f'Ошибка цикла: {error}. Повтор через {delay} с.',
                  file=sys.stderr, flush=True)
        if not args.watch:
            return
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
    replay.add_argument('--start', default=FIRST_ORIGIN)
    replay.add_argument('--end', default=ISSUE_END)
    replay.add_argument('--score-start')
    replay.add_argument('--score-end')
    replay.add_argument('--training-archive', type=Path)
    replay.add_argument('--horizon', choices=['24', '48', 'both'], default='both')
    case = commands.add_parser('run-case', help='Download, validate, replay and report the full case')
    case.add_argument('--archive', type=Path)
    case.add_argument('--raw-dir', type=Path)
    case.add_argument('--training-archive', type=Path)
    case.add_argument('--training-days', type=int, default=14)
    case.add_argument('--workers', type=int, default=6)
    case.add_argument('--first-origin', default=FIRST_ORIGIN)
    case.add_argument('--last-origin', default=LAST_ORIGIN)
    case.add_argument('--score-start', default=SCORE_START)
    case.add_argument('--score-end', default=SCORE_END)
    case.add_argument('--offline', action='store_true')
    case.add_argument('--watch', action='store_true')
    case.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            print(json.dumps(load_history(args.turbine)[1], ensure_ascii=False, indent=2))
        elif args.command == 'run-case':
            if args.interval < 60:
                parser.error('--interval должен быть не менее 60 секунд.')
            case_command(args)
        elif args.command == 'backtest':
            turbines = (1, 2) if args.turbine == 'all' else (int(args.turbine),)
            horizons = (24, 48) if args.horizon == 'both' else (int(args.horizon),)
            score_start = args.score_start or (
                SCORE_START if args.start == FIRST_ORIGIN else args.start)
            score_end = args.score_end or (SCORE_END if args.end == ISSUE_END else args.end)
            reports = []
            for turbine in turbines:
                for horizon in horizons:
                    result = backtest(turbine, args.archive, args.start, args.end,
                                      horizon, score_start=score_start, score_end=score_end,
                                      training_archive_path=args.training_archive)
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
    except (ValueError, OSError) as error:
        parser.exit(2, f'Ошибка: {error}\n')
    except httpx.HTTPError as error:
        parser.exit(2, f'Не удалось получить прогноз погоды: {error}\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
