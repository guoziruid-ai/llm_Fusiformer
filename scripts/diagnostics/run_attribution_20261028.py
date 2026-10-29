"""Fixed configuration of the 2026-10-28 retrospective diagnostic.

For a new execution create a different isolated root and independently rerun
the identity audit before changing the paths below. Never use a production
directory as output. Historical source files are strictly read-only.
"""
import json
from pathlib import Path

if __package__:
    from .compare_latent_fit_scope import Config
    from .attribute_cluster_drift import AttributionConfig, run
else:
    from compare_latent_fit_scope import Config
    from attribute_cluster_drift import AttributionConfig, run

PROJECT = Path('/root/autodl-tmp/llm_fusiformer')
ISOLATION = Path('/root/autodl-tmp/fusiformer_algorithm_attribution_20261028_5Eq9xq')
PROVENANCE = Config(
    latent_path=str(PROJECT / 'runs/fusiformer_latent_eform_clustering/embedding_cache/fusiformer_graph_out.npy'),
    identity_path=str(ISOLATION / 'inputs/identity.csv'),
    evidence_path=str(ISOLATION / 'inputs/evidence.json'),
    output_dir=str(ISOLATION / 'not_used_by_attribution'),
    allow_controlled_fit=False, k=5, l2_normalize=True, seed=42,
    n_init=20, max_iter=500, tol=1e-4,
)
CFG = AttributionConfig(provenance=PROVENANCE, output_dir=str(ISOLATION / 'result'),
                        allow_diagnostic_fit=True)

if __name__ == '__main__':
    summary = run(CFG)
    print(json.dumps({'output_dir': CFG.output_dir, 'model_count': len(summary['models']),
                      'comparison_count': len(summary['comparisons'])}))
