"""Frozen descriptive test check: existing train-only router vs train-only KMeans.

Test has been explored previously: this is not a pristine blind assessment.
Both models are fitted before this script runs and do not use test to fit.
"""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

if __package__:
    from .compare_latent_fit_scope import sha256
    from .run_stopping_audit_20261028 import PROVENANCE
    from .compare_latent_fit_scope import preflight
else:
    from compare_latent_fit_scope import sha256
    from run_stopping_audit_20261028 import PROVENANCE
    from compare_latent_fit_scope import preflight

ROOT = Path('/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ')
OUTPUT = ROOT / 'frozen_test'
REFERENCE = Path('/root/autodl-tmp/llm_fusiformer/runs/grouped3_expert_matscibert/20260910-203626/router/kmeans_train_only_diagnostic.joblib')
REFERENCE_SHA = '254c817dd3ceb0c595900308a412c5500ead8dea399914ade7c8d41bb58dcdd0'
CANDIDATE = ROOT / 'full/kmeans_train.joblib'
CANDIDATE_SHA = '4023c4cbc228b2639a3883cfbc95a7940fb21d7091311c076a6bc1582a5341c3'
SUMMARY = ROOT / 'full/summary.json'
SUMMARY_SHA = 'f99b15b9b2ba61774fe0ff1627ad4881aad129ae5f41891010e14f6493686563'


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    for path, expected in ((REFERENCE, REFERENCE_SHA), (CANDIDATE, CANDIDATE_SHA), (SUMMARY, SUMMARY_SHA)):
        if sha256(path) != expected:
            raise ValueError(f'changed frozen artifact: {path}')
    summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    if summary['models']['kmeans_train']['scope'] != 'train':
        raise ValueError('candidate was not train-only fit')
    vectors, rows, _, hashes = preflight(PROVENANCE)
    split = np.array([r['split'] for r in rows])
    train, test = split == 'train', split == 'test'
    x = normalize(np.asarray(vectors, dtype=np.float32), axis=1).astype(np.float32)
    ref, new = joblib.load(REFERENCE), joblib.load(CANDIDATE)
    a_train = ref.predict(x[train].astype(ref.cluster_centers_.dtype, copy=False))
    b_train = new.predict(x[train].astype(new.cluster_centers_.dtype, copy=False))
    train_matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(train_matrix, (a_train, b_train), 1)
    left, right = linear_sum_assignment(train_matrix, maximize=True)
    mapping = np.empty(5, dtype=int)
    mapping[right] = left
    a = ref.predict(x[test].astype(ref.cluster_centers_.dtype, copy=False))
    raw = new.predict(x[test].astype(new.cluster_centers_.dtype, copy=False))
    b = mapping[raw]
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (a, b), 1)
    xs = x[test].astype(np.float64)
    sse = {}
    for name, model in (('existing_router', ref), ('shared_init_train_kmeans', new)):
        centers = model.cluster_centers_.astype(np.float64)
        sse[name] = float(np.min(((xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1).sum())
    result = {'scope': 'one frozen comparison on previously explored test, descriptive only, no performance evaluation',
              'mapping_fit': 'train only', 'mapping': mapping.tolist(), 'train_matrix': train_matrix.tolist(),
              'test_count': int(test.sum()), 'agreement': float((a == b).mean()),
              'ari': float(adjusted_rand_score(a, b)), 'nmi': float(normalized_mutual_info_score(a, b)),
              'test_migration': matrix.tolist(), 'existing_counts': np.bincount(a, minlength=5).tolist(),
              'candidate_counts_aligned': np.bincount(b, minlength=5).tolist(),
              'test_nearest_center_sse_float64': sse, 'input_hashes': hashes,
              'reference_sha256': REFERENCE_SHA, 'candidate_sha256': CANDIDATE_SHA,
              'fit_summary_sha256': SUMMARY_SHA}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (OUTPUT / 'samples.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'original_index', 'vector_row', 'split', 'existing_cluster',
                         'candidate_raw', 'candidate_aligned', 'changed'])
        for row, i, j, k in zip((r for r in rows if r['split'] == 'test'), a, raw, b):
            writer.writerow([row['sample_id'], row['original_index'], row['vector_row'], 'test',
                             int(i), int(j), int(k), int(i != k)])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
