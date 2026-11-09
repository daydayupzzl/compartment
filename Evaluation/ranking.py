import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

KEY_COLS = ['dataset', 'method', 'data_type']
WEIGHTS = {
    'sc': {'F1': 0.4, 'Correlation': 0.3, 'Count': 0.0, 'ARI': 0.3},
    'bulk': {'F1': 0.4, 'Correlation': 0.3, 'Count': 0.0, 'Strength': 0.3},
}


def scenario_of(path):
    name = os.path.basename(path).lower()
    if name.startswith('sc'):
        return 'sc'
    if name.startswith('bulk'):
        return 'bulk'
    raise ValueError(f'Cannot infer scenario (sc/bulk) from file name: {path}')


def check_input(df, metrics, setting):
    missing_cols = [c for c in KEY_COLS + metrics if c not in df.columns]
    if missing_cols:
        raise ValueError(f'[{setting}] missing columns: {missing_cols}')
    dup = df.duplicated(KEY_COLS)
    if dup.any():
        raise ValueError(f'[{setting}] duplicated rows:\n{df.loc[dup, KEY_COLS]}')
    combos = df.groupby('dataset').size()
    if combos.nunique() != 1:
        raise ValueError(f'[{setting}] unequal number of combinations per dataset:\n{combos}')

    for m in metrics:
        wide = df.pivot_table(index=['method', 'data_type'], columns='dataset', values=m)
        cols = list(wide.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if np.allclose(wide[cols[i]].values, wide[cols[j]].values, equal_nan=True):
                    warnings.warn(f'[{setting}] {m}: datasets {cols[i]} and {cols[j]} have identical values')
        zero_ds = [d for d in cols if np.all(wide[d].values == 0)]
        if zero_ds:
            warnings.warn(f'[{setting}] {m}: all values are 0 in datasets {zero_ds}')


def rank_setting(df, weights, setting='', zero_as_missing=False):
    metrics = list(weights)
    df = df.copy()
    check_input(df, metrics, setting)

    if zero_as_missing:
        for m in metrics:
            if m in ('Strength', 'ARI'):
                all_zero = df.groupby('dataset')[m].transform(lambda s: (s == 0).all())
                df.loc[all_zero, m] = np.nan

    # rank within each dataset (larger is better, ties -> min rank), then average over datasets
    for m in metrics:
        df[f'{m}_rank'] = df.groupby('dataset')[m].rank(method='min', ascending=False)

    summary = df.groupby(['method', 'data_type'])[[f'{m}_rank' for m in metrics]].mean()
    summary.columns = [f'{m}_mean_rank' for m in metrics]
    summary['composite_score'] = sum(summary[f'{m}_mean_rank'] * w for m, w in weights.items())
    summary = summary.sort_values('composite_score', kind='mergesort')
    summary['overall_rank'] = np.arange(1, len(summary) + 1)

    ties = summary['composite_score'].round(9).duplicated(keep=False)
    if ties.any():
        warnings.warn(f'[{setting}] tied composite scores: {summary.index[ties].tolist()}')
    close = summary.index[summary['composite_score'].diff().abs() < 0.05].tolist()
    if close:
        warnings.warn(f'[{setting}] near-tied with the preceding row (|delta| < 0.05): {close}')

    return df, summary.reset_index()


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Weighted rank-based ranking of compartment calling methods')
    parser.add_argument('--inputs', nargs='+', default=None, help='input csv files (default: ranking_data/*.csv)')
    parser.add_argument('--out_dir', default=os.path.join(here, 'ranking_out'), help='output directory')
    parser.add_argument('--zero_as_missing', action='store_true', help='treat all-zero Strength/ARI columns as missing')
    args = parser.parse_args(argv)

    inputs = args.inputs or sorted(glob.glob(os.path.join(here, 'ranking_data', '*.csv')))
    if not inputs:
        raise SystemExit('No input csv found.')
    os.makedirs(args.out_dir, exist_ok=True)

    for path in inputs:
        setting = os.path.splitext(os.path.basename(path))[0]
        weights = WEIGHTS[scenario_of(path)]
        per_dataset, summary = rank_setting(pd.read_csv(path), weights, setting, args.zero_as_missing)
        per_dataset.to_csv(os.path.join(args.out_dir, f'{setting}_per_dataset.csv'), index=False)
        summary.to_csv(os.path.join(args.out_dir, f'{setting}_ranking.csv'), index=False)
        print(f'\n=== {setting}  weights={weights}')
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
