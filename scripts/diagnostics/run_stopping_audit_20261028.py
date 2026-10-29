"""Fixed, bounded retrospective stopping audit; select pilot or full stage."""
import json
import sys
if __package__:
    from .audit_minibatch_stopping import AuditConfig, run
    from .compare_latent_fit_scope import Config
else:
    from audit_minibatch_stopping import AuditConfig, run
    from compare_latent_fit_scope import Config

OLD = '/root/autodl-tmp/fusiformer_algorithm_attribution_20261028_5Eq9xq'
NEW = '/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ'
PROVENANCE = Config(
    latent_path='/root/autodl-tmp/llm_fusiformer/runs/fusiformer_latent_eform_clustering/embedding_cache/fusiformer_graph_out.npy',
    identity_path=OLD + '/inputs/identity.csv', evidence_path=OLD + '/inputs/evidence.json',
    output_dir=NEW + '/unused', allow_controlled_fit=False,
    k=5, l2_normalize=True, seed=42, n_init=20, max_iter=500, tol=1e-4,
)
CFG = AuditConfig(PROVENANCE, NEW, allow_diagnostic_fit=True)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('pilot', 'full'):
        raise SystemExit('expected pilot or full')
    summary = run(CFG, sys.argv[1])
    print(json.dumps({'stage': sys.argv[1], 'models': len(summary['models']),
                      'comparisons': len(summary['comparisons'])}))
