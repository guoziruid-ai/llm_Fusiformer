"""Read-only preflight of the documented historical files; no analysis writes.

The only existing paired V2 prediction file is test. This merely checks its
identity and source; test is not used here to select a scientific method.
For a future approved analysis use a separate explicit Config and fresh output.
"""
import json
from pathlib import Path
from scripts.diagnostics.analyze_frozen_history import Config, preflight

PROJECT = Path('/root/autodl-tmp/llm_fusiformer')
RUN = PROJECT / 'runs/grouped3_expert_matscibert/20260910-203626/router'
CACHE = PROJECT / 'runs/grouped3_expert_matscibert/router_embedding_cache'
AUDIT = Path('/root/autodl-tmp/fusiformer_algorithm_attribution_20261028_5Eq9xq/inputs')
PATHS = {
    'identity': str(AUDIT / 'identity.csv'),
    'identity_evidence': str(AUDIT / 'evidence.json'),
    'assignments': str(PROJECT / 'runs/fusiformer_latent_eform_clustering/20260905-193223/experiment_B_same_K_05/assignments.csv'),
    'predictions': str(PROJECT / 'runs/fusiformer_cluster_prediction_error/20260909-182159/sample_prediction_errors.csv'),
    'mapping': str(RUN / 'difficulty_aware_grouping.json'),
    'router_model': str(RUN / 'kmeans_train_only_diagnostic.joblib'),
    'train_routes': str(RUN / 'train_routes.csv'),
    'train_indices': str(CACHE / 'train_original_indices.npy'),
    'validation_routes': str(RUN / 'val_routes.csv'),
    'validation_indices': str(CACHE / 'val_original_indices.npy'),
}
HASHES = {
    'identity': '2b4a97c8a5dd0c800295b0f2cf65d0e4af9b799a329226c2c9e412e8cdd62da2',
    'identity_evidence': '09d45fcd1f9cc555f3cf860c8fc5738126c0caafa91920b84a09f43e8c58ed62',
    'assignments': '958f29dcabde9af040b2d60121870de76a6d18f5959e044b9bbd3b71efc0d5a0',
    'predictions': '743d2ac79f8a5e12a8a1b7672786f572cef096f91044ba1c499c80990d7cea9f',
    'mapping': '24cb9615225fd4c3b8b084b4e09564a8b20126ae8c26262b94f3e38f13d12804',
    'router_model': '254c817dd3ceb0c595900308a412c5500ead8dea399914ade7c8d41bb58dcdd0',
    'train_routes': 'bf1ce88116c0b63af77a0354ace318a640314bdacdfe449e0b7134f9ff979060',
    'train_indices': '2861f263232cbd4776deaf27e63604122afb8d59843e35e0b3ec52f55f56a67d',
    'validation_routes': 'fb449efcabee0781f146254d02c8069c6d7a0e7c24ec4a20d80c4c48c48fc371',
    'validation_indices': '3b54810fcc816d4e93ccb28cec1c5120f3610f13618a277b0c61bbfe2c22fc04',
}
CFG = Config(paths=PATHS, sha256=HASHES, prediction_split='test',
             output_dir='', allow_write=False, plot=False, require_prediction_routes=False)

if __name__ == '__main__':
    print(json.dumps(preflight(CFG), ensure_ascii=False))
