import argparse
import math
import re
import subprocess
import sys


RUNTIME_MARKERS = (
    'Traceback',
    'RuntimeError',
    'CUDA out of memory',
    'AssertionError',
    'ValueError',
    'TypeError',
)

EPOCH_RE = re.compile(r'Epoch\(train\)\s*\[(\d+)\]\[\s*(\d+)/(\d+)\]')
VALUE_PATTERNS = {
    'loss': re.compile(r'\bloss:\s*([0-9.eE+-]+|inf|nan)\b'),
    'grad_norm': re.compile(r'\bgrad_norm:\s*([0-9.eE+-]+|inf|nan)\b'),
    'mean_u_all': re.compile(r'\bmean_u_all:\s*([0-9.eE+-]+|inf|nan)\b'),
    'loss_evi_total': re.compile(r'\bloss_evi_total:\s*([0-9.eE+-]+|inf|nan)\b'),
    'evi_raw_img': re.compile(r'\bevi_raw_img:\s*([0-9.eE+-]+|inf|nan)\b'),
    'evi_raw_lidar': re.compile(r'\bevi_raw_lidar:\s*([0-9.eE+-]+|inf|nan)\b'),
    'evi_raw_radar': re.compile(r'\bevi_raw_radar:\s*([0-9.eE+-]+|inf|nan)\b'),
}


def parse_float(text: str) -> float:
    text = text.lower()
    if text == 'inf':
        return float('inf')
    if text == 'nan':
        return float('nan')
    return float(text)


def analyze_line(line: str):
    if any(marker in line for marker in RUNTIME_MARKERS):
        return 'ALERT runtime', line

    match = EPOCH_RE.search(line)
    if not match:
        return None

    epoch = int(match.group(1))
    iteration = int(match.group(2))
    total = int(match.group(3))
    values = {}
    for name, pattern in VALUE_PATTERNS.items():
        value_match = pattern.search(line)
        if value_match:
            values[name] = parse_float(value_match.group(1))

    alerts = []
    for name in ('loss', 'mean_u_all', 'loss_evi_total'):
        if name in values and (math.isnan(values[name]) or math.isinf(values[name])):
            alerts.append(f'{name}={values[name]}')

    if 'mean_u_all' in values and values['mean_u_all'] > 0.95:
        alerts.append(f"mean_u_all={values['mean_u_all']:.4f}")

    for name in ('evi_raw_img', 'evi_raw_lidar', 'evi_raw_radar'):
        if name in values and values[name] < 0.1:
            alerts.append(f"{name}={values[name]:.4f}")

    if alerts:
        return f"ALERT epoch={epoch} iter={iteration}/{total} :: {'; '.join(alerts)}", line

    if epoch == 1 and iteration == total:
        return f'EPOCH1_DONE epoch={epoch} iter={iteration}/{total}', line

    if epoch > 1:
        return f'EPOCH1_DONE observed_epoch={epoch}', line

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    args = parser.parse_args()

    proc = subprocess.Popen(
        ['tail', '-n', '0', '-F', args.logfile],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip('\n')
            result = analyze_line(line)
            if result is None:
                continue
            print(result[0], flush=True)
            print(result[1], flush=True)
            proc.terminate()
            return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 1


if __name__ == '__main__':
    sys.exit(main())