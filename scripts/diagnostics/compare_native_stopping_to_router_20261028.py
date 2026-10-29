"""Train-matched audit of old/new MiniBatch stops against frozen train KMeans."""
import csv
import json
from pathlib import Path
import numpy as np
if __package__:
    from .attribute_cluster_drift import compare
    from .compare_latent_fit_scope import sha256
else:
    from attribute_cluster_drift import compare
    from compare_latent_fit_scope import sha256

ROOT = Path('/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ')
FULL = ROOT / 'full/sample_labels.csv'
NATIVE = ROOT / 'native_counterfactual/sample_labels.csv'
OUTPUT = ROOT / 'native_vs_train_router'
EXPECTED = {FULL: '588fc1f7bd4dafd47918c403f61f51ae24814b7516c8617bd536241e5638c5e6',
            NATIVE: '4a6f8b9642a72b78eb35a63551f47af1b8783b4087a190318b2f01c459f96688'}

def read(path, labels):
    rows = {}
    with path.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            v = int(r['vector_row'])
            if v in rows or r['split'] not in ('train', 'validation'):
                raise ValueError('duplicate index or non-train/validation row')
            rows[v] = (r['sample_id'], int(r['original_index']), r['split'],
                       *[int(r[name]) for name in labels])
    return rows

def main():
    if OUTPUT.exists():
        raise FileExistsError('output exists')
    for path, digest in EXPECTED.items():
        if sha256(path) != digest:
            raise ValueError(f'changed input {path}')
    left, right = read(FULL, ['kmeans_train']), read(NATIVE, ['baseline', 'bounded'])
    if left.keys() != right.keys():
        raise ValueError('missing or extra vector identities')
    ids = sorted(left)
    if any(left[i][:3] != right[i][:3] for i in ids):
        raise ValueError('mp_id, original index or split mismatch')
    labels = {'kmeans_train': np.array([left[i][3] for i in ids]),
              'native_early': np.array([right[i][3] for i in ids]),
              'native_bounded': np.array([right[i][4] for i in ids])}
    masks = {s: np.array([left[i][2] == s for i in ids]) for s in ('train', 'validation')}
    results = {name: compare(labels['kmeans_train'], labels[name], masks['train'], masks, 5)
               for name in ('native_early', 'native_bounded')}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    report = {'matching': 'train-only; applied to validation', 'input_hashes': {str(p): h for p, h in EXPECTED.items()},
              'comparisons': results}
    (OUTPUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (OUTPUT / 'sample_migrations.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'original_index', 'vector_row', 'split', 'kmeans_train',
                         'native_early_aligned', 'native_bounded_aligned', 'early_changed', 'bounded_changed'])
        maps = {name: np.asarray(results[name]['candidate_to_reference']) for name in results}
        for pos, i in enumerate(ids):
            e, b, k = maps['native_early'][labels['native_early'][pos]], maps['native_bounded'][labels['native_bounded'][pos]], labels['kmeans_train'][pos]
            writer.writerow([left[i][0], left[i][1], i, left[i][2], int(k), int(e), int(b), int(k != e), int(k != b)])
    print(json.dumps(report))

if __name__ == '__main__':
    main()
