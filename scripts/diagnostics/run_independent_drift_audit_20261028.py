"""Fixed configuration for this read-only historical metric recheck."""
import json
if __package__:
    from .independent_cluster_metrics_audit import AuditConfig, run
else:
    from independent_cluster_metrics_audit import AuditConfig, run

CFG = AuditConfig(
    source='/root/autodl-tmp/fusiformer_algorithm_attribution_20261028_5Eq9xq',
    output='/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ/independent',
    enabled=True,
)

if __name__ == '__main__':
    report = run(CFG)
    print(json.dumps({'comparisons_verified': report['saved_comparisons_verified'],
                      'raw_match': report['raw_historical_match'],
                      'counts': report['counts'],
                      'manifest_files_verified': report['manifest_files_verified']}))
