"""One predeclared fresh target: baseline plus five fixed bilevel lambdas."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import load_state, write_json
from experiments.run_bilevel_trials import BilevelTrials, SliceExpired, make_config
from experiments.run_bilevel_overnight import slice_outcome


def validate_protocol(protocol):
    expected_optimizer = dict(learning_rate=.1, momentum=.9, nesterov=True,
        weight_decay=.0005, clipping=False, warmup=False, adaptive_lambda=False, update_rescaling=False)
    if (protocol['trial_count'] != 1 or protocol['case_keys'] != [2000000000] or
            protocol['victim_seeds'] != [18001] or protocol['surrogate_seed'] != 314159 or
            protocol['lambdas'] != [1e-10, 1e-8, 1e-7, 1e-6, 1e-5] or
            protocol['victim_models'] != 6 or protocol['new_victim_trainings'] != 6 or
            protocol['reused_undefended_models'] != 0 or protocol['additional_clean_surrogates'] != 0 or
            protocol['dataset'] != 'CIFAR10' or protocol['model'] != 'ResNet18' or
            protocol['train_images'] != 50000 or protocol['test_images'] != 10000 or
            protocol['epochs'] != 40 or protocol['batch_size'] != 128 or
            protocol['attack'] != dict(recipe='gradient-matching', budget=.01,
                epsilon_pixel_units=16., iterations=250, restarts=8, poison_batch=128, loss='similarity') or
            protocol['optimizer'] != expected_optimizer or
            protocol['defense']['radius_pixel_units'] != 16. or
            protocol['defense']['inner_steps'] != 3 or
            protocol['defense']['inner_step_pixel_units'] != 32/3 or
            protocol['defense']['gradient_chunk_size'] != 8 or
            protocol['defense']['regularizer_samples_per_batch'] != 8):
        raise ValueError('Require the single new target, seed 18001, baseline and five declared lambdas.')


def verify_surrogate(protocol, source):
    for relative, expected in protocol['frozen_source_files'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen clean surrogate source changed: '+relative)
    manifest = json.loads((source/'manifest.json').read_text())
    result = json.loads((source/'runs/surrogate.json').read_text())
    expected = dict(dataset='CIFAR10', model='ResNet18', train_size=None,
                    epochs=40, batch_size=128, surrogate_seed=314159, precision='fp32')
    if any(manifest['config'].get(k) != v for k, v in expected.items()):
        raise RuntimeError('Clean surrogate configuration mismatch.')
    if (result.get('name') != 'surrogate' or result.get('seed') != 314159 or
            result.get('status') != 'complete' or result.get('poisoned') is not False or
            result.get('clean_test_accuracy', 0) < .85):
        raise RuntimeError('Clean surrogate identity or accuracy mismatch.')
    return manifest, result


class NewTargetTrials(BilevelTrials):
    def run_new_target(self):
        case = self.cases[0]
        if case['target_id'] in self.protocol['excluded_target_ids'] or len(case['poison_ids']) != 500:
            raise RuntimeError('The declared case must select a new target and exactly 500 poisons.')
        identity_path = self.out/'case.json'
        identity = dict(case=case, classes=list(self.full_validset.classes),
                        selection='Predeclared case key; retained regardless of attack or defense outcome.')
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError('Target or poison row identities changed between scheduler slices.')
        write_json(identity_path, identity)
        self.manifest['confidence'] = dict(log_base='e',
            computation='FP64 log_softmax of the model FP32 logits; probabilities are exp(log_probability).',
            false_class='The predeclared attacker-intended class, not the highest-scoring alternative.',
            classes=list(self.full_validset.classes),
            observations='Initialization, first 32 updates, checkpoints, each epoch, final evaluation. Observations do not tune training.')
        write_json(self.out/'manifest.json', self.manifest)
        self.summarize_trials()
        poison_path = self.poison_path(case)
        if not poison_path.exists() and self.out_of_time(margin=1800):
            raise SliceExpired()
        write_json(self.out/'status.json', dict(phase='brewing', case_key=case['key'], target_id=case['target_id']))
        delta = self.brew(case, load_state(self.source/'models/surrogate.pt'))
        poison = load_state(poison_path)
        digest = hashlib.sha256(poison_path.read_bytes()).hexdigest()
        info = json.loads(poison_path.with_suffix('.json').read_text())
        if (int(poison['target_id']) != case['target_id'] or
                poison['poison_ids'].tolist() != case['poison_ids'] or
                digest != info['sha256'] or not torch.isfinite(delta).all()):
            raise RuntimeError('Saved new poison identity or hash mismatch.')
        verified_path = self.out/'inputs_verified.json'
        if verified_path.exists() and json.loads(verified_path.read_text())['poison_sha256'] != digest:
            raise RuntimeError('New poison changed between scheduler slices.')
        write_json(verified_path, dict(frozen_source_files=self.protocol['frozen_source_files'],
            source_directory=str(self.source), source_use='Reuse clean surrogate weights solely for new poison brewing.',
            case_key=case['key'], target_id=case['target_id'], comparison_seed=18001,
            poison_sha256=digest, baseline_reused=False, new_poison_generation=True,
            new_poison_cases=1, additional_clean_surrogates=0))
        conditions = [('standard', None)]+[('lambda-'+str(w), w) for w in self.protocol['lambdas']]
        for label, weight in conditions:
            self.train_condition('trial-0-'+label, 18001, weight, case, delta)
            self.summarize_trials()
        final = self.summarize_trials()
        write_json(self.out/'status.json', dict(phase='complete' if final['all_succeeded'] else 'complete_with_failures',
            attempted=final['attempted'], expected=6, new_defended_trainings=5,
            new_undefended_trainings=1, cached_baselines=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='experiments/protocols/bilevel-new-target-2000000000.json')
    parser.add_argument('--output', default='artifacts/bilevel-new-target-2000000000')
    parser.add_argument('--data', default='~/data')
    parser.add_argument('--slice-seconds', type=float, default=6000.)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    validate_protocol(protocol)
    root = Path(args.output)
    hard_stop = datetime.fromisoformat(protocol['hard_stop_utc']).timestamp()
    if time.time() >= hard_stop-180:
        write_json(root/'status.json', dict(phase='deadline_reached', hard_stop_utc=protocol['hard_stop_utc']))
        return 0
    try:
        source = Path(protocol['source_directory']).expanduser().resolve()
        source_manifest, _ = verify_surrogate(protocol, source)
        if torch.__version__ != source_manifest['torch']:
            raise RuntimeError('PyTorch differs from the frozen surrogate environment.')
        config = make_config(args, protocol)
        config.regularizer_samples = 8
        config.gradient_diagnostics = True
        config.update_diagnostics = True
        config.validate_buffers = True
        config.log_target_confidences = True
        config.save_failure_checkpoint = True
        experiment = NewTargetTrials(config, protocol, args.slice_seconds)
        experiment.source = source
        experiment.deadline = min(experiment.deadline, time.monotonic()+max(0, hard_stop-time.time()))
        experiment.run_new_target()
    except SliceExpired:
        phase, code = slice_outcome(hard_stop, time.time())
        write_json(root/'status.json', dict(phase=phase, hard_stop_utc=protocol['hard_stop_utc'],
            reason='Exact checkpoint saved; continue only before the morning cutoff.'))
        return code
    except BaseException:
        write_json(root/'status.json', dict(phase='failed', traceback=traceback.format_exc()))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
