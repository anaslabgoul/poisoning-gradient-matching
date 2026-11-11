"""One matched paper-setting ResNet18 trial, bounded to one night.

All network parameters participate. Eight randomly selected penalty images per
128-image classification batch make the per-example regularizer affordable.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import write_json
from experiments.run_bilevel_trials import BilevelTrials, SliceExpired, make_config


def slice_outcome(hard_stop, now):
    """A too-short final slice must stop instead of repeatedly resubmitting."""
    if hard_stop-now <= 1800:
        return 'deadline_reached', 0
    return 'checkpointed', 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='experiments/protocols/bilevel-resnet18-overnight-once.json')
    parser.add_argument('--output', default='artifacts/bilevel-overnight')
    parser.add_argument('--data', default='~/data')
    parser.add_argument('--slice-seconds', type=float, default=6000.)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if (protocol['trial_count']!=1 or protocol['victim_models']!=2 or
        protocol['model']!='ResNet18' or protocol['dataset']!='CIFAR10' or
        protocol['lambdas']!=[.001] or protocol['train_images']!=50000 or
        protocol['defense']['regularizer_samples_per_batch']!=8):
        raise ValueError('Require exactly the approved one-trial overnight protocol.')
    root = Path(args.output)
    hard_stop = datetime.fromisoformat(protocol['hard_stop_utc']).timestamp()
    if time.time() >= hard_stop-180:
        write_json(root/'status.json', dict(phase='deadline_reached', hard_stop_utc=protocol['hard_stop_utc'],
            reason='Overnight deadline reached. No new training was started.'))
        return 0
    config = make_config(args, protocol)
    config.regularizer_samples = protocol['defense']['regularizer_samples_per_batch']
    config.hard_stop_utc = protocol['hard_stop_utc']
    try:
        experiment = BilevelTrials(config, protocol, args.slice_seconds)
        experiment.deadline = min(experiment.deadline, time.monotonic()+max(0, hard_stop-time.time()))
        experiment.run_trials()
    except SliceExpired:
        phase, code = slice_outcome(hard_stop, time.time())
        write_json(root/'status.json', dict(phase=phase, hard_stop_utc=protocol['hard_stop_utc'],
            reason='Checkpointed. Morning cutoff prevents further continuation.' if code==0 else
                   'Checkpointed. Continue within the same night.'))
        return code
    except BaseException:
        write_json(root/'status.json', dict(phase='failed', traceback=traceback.format_exc()))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
