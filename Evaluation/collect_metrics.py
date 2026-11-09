import os
import re
import glob
import argparse
import pandas as pd

METHOD_NAMES = {'pca': 'PCA', 'cpg': 'CPG', 'higashi': 'Ref-PCA', 'scghost': 'scGHOST'}
DATATYPE_NAMES = {'raw': 'Raw', 'schicluster': 'scHiCluster', 'higashi': 'Higashi'}
HERE = os.path.dirname(os.path.abspath(__file__))


def _safe(name):
    return re.sub(r'[\\/:*?"<>|\s]', '_', str(name))


def export_notebook_metrics(results, dataset, cell_type, setting, out_dir=None, strength=None):
    out_dir = out_dir or os.path.join(HERE, 'metrics_out')
    rows = []
    for m, per_type in results.items():
        for dt, v in per_type.items():
            row = {'dataset': dataset, 'cell_type': cell_type,
                   'method': METHOD_NAMES.get(m, m), 'data_type': DATATYPE_NAMES.get(dt, dt),
                   'F1': v['f1'], 'Correlation': v['corr'], 'Count': v['boundary_count']}
            if strength is not None:
                row['Strength'] = strength[m][dt]
            rows.append(row)
    path = os.path.join(out_dir, setting, f'{_safe(dataset)}__{_safe(cell_type)}.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def collect(setting, metrics_dir, cluster_csv=None):
    files = sorted(glob.glob(os.path.join(metrics_dir, setting, '*.csv')))
    if not files:
        raise FileNotFoundError(f'no metric files in {os.path.join(metrics_dir, setting)}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = ['dataset', 'method', 'data_type']

    # one representative cell type per dataset
    cts = df.groupby('dataset')['cell_type'].unique()
    multi = {d: list(c) for d, c in cts.items() if len(c) > 1}
    if multi:
        raise ValueError(f'one representative cell type per dataset is expected, found several: {multi}')

    if setting.startswith('sc'):
        if cluster_csv is None:
            raise ValueError('single-cell settings need --cluster (ARI)')
        cl = pd.read_csv(cluster_csv)
        for c in keys:
            cl[c] = cl[c].astype(str).str.strip()
        df = df.merge(cl[keys + ['ARI']], on=keys, how='left', validate='one_to_one')
        miss = df[df['ARI'].isna()]
        if len(miss):
            raise ValueError(f'ARI missing for:\n{miss[keys].to_string(index=False)}')
        cols = keys + ['cell_type', 'F1', 'Correlation', 'Count', 'ARI']
    else:
        if 'Strength' not in df.columns or df['Strength'].isna().any():
            raise ValueError('bulk settings need a complete Strength column')
        cols = keys + ['cell_type', 'F1', 'Correlation', 'Count', 'Strength']
    return df[cols].sort_values(keys).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Collect per-dataset metrics into ranking_data/<setting>.csv')
    parser.add_argument('--setting', required=True, help='sc_100k / sc_500k / bulk300_100k / bulk300_500k')
    parser.add_argument('--metrics_dir', default=os.path.join(HERE, 'metrics_out'), help='metrics exported by get_result.ipynb')
    parser.add_argument('--cluster', default=None, help='clustering csv (required for single-cell settings)')
    parser.add_argument('--out', default=None, help='output csv (default: ranking_data/<setting>.csv)')
    args = parser.parse_args(argv)

    out = args.out or os.path.join(HERE, 'ranking_data', f'{args.setting}.csv')
    df = collect(args.setting, args.metrics_dir, args.cluster)
    df.to_csv(out, index=False)
    print(f'wrote {out} ({len(df)} rows, {df.dataset.nunique()} datasets)')


if __name__ == '__main__':
    main()
