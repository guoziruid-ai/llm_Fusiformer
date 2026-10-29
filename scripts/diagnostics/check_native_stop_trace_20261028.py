"""Check whether initial candidates and minibatch traces match up to stop."""
import json
import math
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp/fusiformer_drift_audit_20261028_3FWKIZ')
FILES = [ROOT / 'pilot/historical_parameters_all.verbose.log',
         ROOT / 'native_counterfactual/bounded.verbose.log']

def parse(path):
    inits, steps = [], {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith(('Init ', 'Inertia for init ')):
            inits.append(line)
        match = re.match(r'Minibatch step (\d+)/\d+: mean batch inertia: ([^,]+)(?:, ewa inertia: (.+))?', line)
        if match:
            steps[int(match.group(1))] = [float(x) for x in match.groups()[1:] if x is not None]
    return inits, steps

def main():
    (a, x), (b, y) = [parse(p) for p in FILES]
    if a != b or len(a) != 40 or set(range(1, 46)) - x.keys() or set(range(1, 46)) - y.keys():
        raise ValueError('initialization or first 45 logged steps do not match')
    error = max(abs(i - j) for step in range(1, 46) for i, j in zip(x[step], y[step]))
    print(json.dumps({'initialization_lines_equal': True, 'initialization_lines': len(a),
                      'steps_compared': 45, 'max_abs_logged_inertia_difference': error,
                      'matches_1e_minus_5': math.isclose(error, 0.0, abs_tol=1e-5)}))

if __name__ == '__main__':
    main()
