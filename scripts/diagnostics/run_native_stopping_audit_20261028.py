"""Fixed historical-parameter early-stop counterfactual, train/val only."""
import json
if __package__:
    from .audit_native_early_stop import NativeStopConfig, run
    from .run_stopping_audit_20261028 import PROVENANCE, NEW
else:
    from audit_native_early_stop import NativeStopConfig, run
    from run_stopping_audit_20261028 import PROVENANCE, NEW

CFG = NativeStopConfig(
    provenance=PROVENANCE,
    baseline_model=NEW + '/pilot/historical_parameters_all.joblib',
    baseline_model_sha256='fc32a3fa94c0026bf7249d83581995f3bcc0331d2f4cc90e6785b36fc27e6a5a',
    baseline_summary=NEW + '/pilot/summary.json',
    baseline_summary_sha256='a76afcbed1d637b29af47c623876001e7a13cb82e0fbb866d6dd045b848f3d8b',
    output_dir=NEW + '/native_counterfactual', allow_diagnostic_fit=True,
)

if __name__ == '__main__':
    result = run(CFG)
    print(json.dumps({'steps': result['bounded']['n_steps'],
                      'baseline_raw_matches': result['historical_baseline_raw_match_counts']}))
