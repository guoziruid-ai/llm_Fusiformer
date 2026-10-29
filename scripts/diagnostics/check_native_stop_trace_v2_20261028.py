"""Numerically compare candidate inertias and first 45 batch trace values."""
import json
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ')
FILES = [ROOT / 'pilot/historical_parameters_all.verbose.log',
         ROOT / 'native_counterfactual/bounded.verbose.log']

def parse(path):
    inits, steps = {}, {}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'Inertia for init (\d+)/20: (.+)', line)
        if m:
            inits[int(m.group(1))] = float(m.group(2))
        m = re.match(r'Minibatch step (\d+)/\d+: mean batch inertia: ([^,]+)(?:, ewa inertia: (.+))?', line)
        if m:
            steps[int(m.group(1))] = [float(x) for x in m.groups()[1:] if x is not None]
    return inits, steps

def main():
    (a, x), (b, y) = [parse(p) for p in FILES]
    if a.keys() != b.keys() or len(a) != 20 or set(range(1, 46)) - x.keys() or set(range(1, 46)) - y.keys():
        raise ValueError('missing initialization or first 45 steps')
    delta_init = max(abs(a[k] - b[k]) for k in a)
    delta_steps = max(abs(i - j) for step in range(1, 46) for i, j in zip(x[step], y[step]))
    print(json.dumps({'initialization_candidate_count': len(a), 'best_init_indices': [min(a, key=a.get), min(b, key=b.get)],
                      'max_abs_init_inertia_difference': delta_init, 'steps_compared': 45,
                      'max_abs_logged_batch_inertia_difference': delta_steps,
                      'caveat': 'numeric trace similarities do not prove bitwise identical internal states'}))

if __name__ == '__main__':
    main()
