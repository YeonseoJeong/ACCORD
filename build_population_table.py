#!/usr/bin/env python3
"""Convert *original gen_eval.py* JSON episode results into paper plotting CSVs.

Expected JSON files in --json-dir:
  allon_N{N}.json, rule_N{N}.json, fixed_N{N}.json,
  dynamic_N{N}.json, esonly_N{N}.json

Outputs in --out:
  population_episodes.csv : 150 episode rows per method/UE count
  population_table.csv    : method/UE means and episode standard deviations

Use as input for the existing plot_population_paper.py with --csv population_table.csv.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS = (
    ('ACCORD (fixed-100)', 'fixed'),
    ('ACCORD (dynamic)', 'dynamic'),
    ('ES-only (dynamic)', 'esonly'),
    ('Rule-based', 'rule'),
    ('All-on', 'allon'),
)
METRICS = ('Avg_Active_BS', 'Service_Degradation', 'Energy_Saving_pct', 'Handover_Rate')


def read_json(folder, key, n, episodes):
    path = folder / f'{key}_N{n}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run the corresponding gen_eval command first.')
    data = json.loads(path.read_text(encoding='utf-8'))
    for field in ('E_sum', 'V', 'n_on', 'H'):
        if field not in data:
            raise ValueError(f'{path}: missing {field}')
        a = np.asarray(data[field], dtype=float)
        if a.shape != (episodes,) or not np.isfinite(a).all():
            raise ValueError(f'{path}: {field} must be {episodes} finite values; got {a.shape}')
        data[field] = a
    return data


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ns', type=int, nargs='+', default=[50, 75, 100, 125, 150])
    parser.add_argument('--episodes', type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw, summary = [], []
    for n in args.ns:
        allon = read_json(args.json_dir, 'allon', n, args.episodes)
        E0 = allon['E_sum']
        if np.any(E0 <= 0):
            raise ValueError(f'All-on energy must be positive for N={n}')
        for label, key in METHODS:
            data = allon if key == 'allon' else read_json(args.json_dir, key, n, args.episodes)
            values = {
                'Avg_Active_BS': data['n_on'],
                'Service_Degradation': data['V'],
                'Energy_Saving_pct': 100.0 * (1.0 - data['E_sum'] / E0),
                'Handover_Rate': data['H'],
            }
            for ep in range(args.episodes):
                raw.append(dict(Method=label, UEs=n, Episode=ep,
                                **{metric: float(values[metric][ep]) for metric in METRICS}))
            summary.append(dict(Method=label, UEs=n,
                                **{metric: float(values[metric].mean()) for metric in METRICS},
                                **{metric + '_std': float(values[metric].std(ddof=1))
                                   for metric in METRICS}))
            print(f'N={n:3d} {label:21s}  ES={values["Energy_Saving_pct"].mean():6.2f}% '
                  f'V={values["Service_Degradation"].mean():.4f} '
                  f'on={values["Avg_Active_BS"].mean():.2f}')

    write_csv(args.out / 'population_episodes.csv', raw,
              ['Method', 'UEs', 'Episode', *METRICS])
    write_csv(args.out / 'population_table.csv', summary,
              ['Method', 'UEs', *METRICS, *(m + '_std' for m in METRICS)])

if __name__ == '__main__':
    main()
