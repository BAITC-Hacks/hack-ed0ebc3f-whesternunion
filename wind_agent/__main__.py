import argparse
import json
import time
from pathlib import Path

import httpx

from .data import load_history
from .agent import explain
from .service import backtest, run_forecast, save_result


def main():
    parser = argparse.ArgumentParser(description='WindPilot: local forecasting tools')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--turbine', type=int, choices=[1, 2], default=1)
    for name in ['forecast', 'watch']:
        command = commands.add_parser(name)
        command.add_argument('--turbine', type=int, choices=[1, 2], default=1)
        command.add_argument('--horizon', type=int, choices=[24, 48], default=48)
        command.add_argument('--mode', choices=['baseline', 'archive', 'live'], default='baseline')
        command.add_argument('--origin')
        command.add_argument('--archive', type=Path)
        command.add_argument('--ai', action='store_true')
        if name == 'watch':
            command.add_argument('--interval', type=int, default=3600)
    replay = commands.add_parser('backtest')
    replay.add_argument('--turbine', type=int, choices=[1, 2], default=1)
    replay.add_argument('--archive', type=Path, required=True)
    replay.add_argument('--start', default='2026-02-01T00:00:00+05:00')
    replay.add_argument('--end', default='2026-03-01T00:00:00+05:00')
    replay.add_argument('--horizon', type=int, choices=[24, 48], default=24)
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            print(json.dumps(load_history(args.turbine)[1], ensure_ascii=False, indent=2))
        elif args.command == 'backtest':
            result = backtest(args.turbine, args.archive, args.start, args.end, args.horizon)
            print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}, ensure_ascii=False, indent=2))
        else:
            if args.command == 'watch' and args.interval < 60:
                parser.error('--interval должен быть не менее 60 секунд.')
            previous = None
            while True:
                watching = args.command == 'watch'
                result = run_forecast(args.turbine, args.horizon, args.origin,
                                      args.mode, args.archive, args.ai and not watching,
                                      persist=not watching)
                changed = result['input_fingerprint'] != previous
                if watching and changed:
                    if args.ai:
                        result['ai'] = explain(result)
                    save_result(result)
                print(json.dumps({'id': result['id'], 'changed': changed,
                                  'summary': result['summary'], 'ai': result['ai']}, ensure_ascii=False, indent=2), flush=True)
                previous = result['input_fingerprint']
                if args.command != 'watch':
                    break
                time.sleep(args.interval)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Ошибка: {error}\n')
    except httpx.HTTPError:
        parser.exit(2, 'Не удалось получить прогноз погоды. Проверьте доступ к Open-Meteo.\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
