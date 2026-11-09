"""Reproducible, resumable CIFAR-10/ResNet18 fixed-poison defense comparison.

Uses the repository's model, normalized data, differentiable augmentation,
optimizer/scheduler and gradient-matching attack. Training tensors are held on
the GPU to avoid repeated image decoding on a shared network home directory.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forest
from forest.victims.defenses import METHODS, paired_defense_loss
from forest.victims.models import get_model
from forest.victims.optimization_strategy import training_strategy
from forest.victims.training import get_optimizers


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def criterion(model, outputs, labels):
    return F.cross_entropy(outputs.float(), labels)


class Comparison:
    def __init__(self, config):
        if not torch.cuda.is_available():
            raise RuntimeError('This experiment runner requires a CUDA GPU.')
        self.config = config
        self.out = Path(config.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(config.threads)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device('cuda:0')
        self.setup = dict(device=self.device, dtype=torch.float32, non_blocking=True)
        self.args = forest.options().parse_args([])
        self.args.dataset = getattr(config, 'dataset', 'CIFAR10')
        self.model_name = getattr(config, 'model', 'ResNet18')
        self.args.net = [self.model_name]
        self.args.data_path = str(Path(config.data).expanduser())
        self.args.epochs = config.epochs
        self.args.attackiter = config.attack_iterations
        self.args.restarts = config.restarts
        self.args.pbatch = config.poison_batch
        self.args.poisonkey = str(config.cases[0])
        self.args.modelkey = config.surrogate_seed
        self.args.budget = config.budget
        self.args.eps = config.eps
        self.args.cache_dataset = False
        self.kettle = forest.Kettle(self.args, batch_size=128, augmentations='default', setup=self.setup)
        self.dm, self.ds = self.kettle.dm, self.kettle.ds
        train = self.kettle.trainset
        def image_tensor(data):
            tensor = torch.as_tensor(data, device=self.device).float()/255
            return tensor[:, None] if tensor.ndim == 3 else tensor.permute(0, 3, 1, 2)
        self.images = image_tensor(train.data)
        self.images = ((self.images - self.dm) / self.ds).contiguous(memory_format=torch.channels_last)
        self.labels = torch.tensor(train.targets, device=self.device, dtype=torch.long)
        # Keep the full un-subsetted test data. The same target union is excluded
        # from accuracy for every clean and poisoned run, including all methods.
        test = self.kettle.validset.dataset
        self.full_validset = test
        self.test_images = image_tensor(test.data)
        self.test_images = ((self.test_images - self.dm) / self.ds).contiguous(memory_format=torch.channels_last)
        self.test_labels = torch.tensor(test.targets, device=self.device, dtype=torch.long)
        self.cases = []
        for key in config.cases:
            self.args.poisonkey = str(key)
            self.kettle.validset = self.full_validset
            self.kettle.random_construction()
            self.cases.append(dict(key=key, target_id=int(self.kettle.target_ids[0]),
                                   true_label=int(self.kettle.poison_setup['target_class']),
                                   intended_label=int(self.kettle.poison_setup['intended_class'][0]),
                                   poison_ids=self.kettle.poison_ids.tolist()))
        self.accuracy_mask = torch.ones(len(self.test_labels), dtype=torch.bool, device=self.device)
        self.accuracy_mask[[case['target_id'] for case in self.cases]] = False
        self.manifest = dict(config=vars(config), cases=self.cases,
                             dataset=self.args.dataset, model=self.model_name,
                             evaluation='fixed standard-gradient-matching poisons; not defense-aware',
                             torch=torch.__version__, torchvision=__import__('torchvision').__version__,
                             gpu=torch.cuda.get_device_name(), tf32=True,
                             normalization=dict(mean=train.data_mean, std=train.data_std),
                             n_train=len(train), n_test_accuracy=int(self.accuracy_mask.sum()),
                             batchnorm='paired views share a single training forward; one running-stat update',
                             gradient_penalty='mean per-example squared Frobenius norm; final weight only; no detach',
                             git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        manifest_path = self.out / 'manifest.json'
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old['config'] != self.manifest['config'] or old['git_commit'] != self.manifest['git_commit']:
                raise RuntimeError('Output belongs to a different configuration/commit. Use a new output directory.')
        else:
            write_json(manifest_path, self.manifest)

    def autocast(self):
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if self.config.precision == 'bf16' else nullcontext())

    def evaluate(self, model):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.
        predictions, probabilities = [], []
        with torch.no_grad():
            for start in range(0, len(self.test_labels), 256):
                stop = start + 256
                with self.autocast():
                    outputs = model(self.test_images[start:stop]).float()
                labels = self.test_labels[start:stop]
                mask = self.accuracy_mask[start:stop]
                correct += int((outputs.argmax(1)[mask] == labels[mask]).sum())
                total += int(mask.sum())
                loss_sum += float(F.cross_entropy(outputs[mask], labels[mask], reduction='sum'))
                predictions.extend(outputs.argmax(1).tolist())
                probabilities.extend(outputs.softmax(1).tolist())
        targets = [dict(case_key=case['key'], target_id=case['target_id'],
                        true_label=case['true_label'], intended_label=case['intended_label'],
                        predicted_label=predictions[case['target_id']],
                        attack_success=int(predictions[case['target_id']] == case['intended_label']),
                        target_correct=int(predictions[case['target_id']] == case['true_label']),
                        intended_probability=probabilities[case['target_id']][case['intended_label']])
                   for case in self.cases]
        label_list = self.test_labels.tolist()
        for target in targets:
            group = [i for i, label in enumerate(label_list) if label == target['true_label']]
            unseen = [i for i in group if i != target['target_id']]
            target['group_size'] = len(group)
            target['group_intended_count'] = sum(predictions[i] == target['intended_label'] for i in group)
            target['group_intended_rate'] = target['group_intended_count']/len(group)
            target['unseen_group_size'] = len(unseen)
            target['unseen_group_intended_rate'] = sum(predictions[i] == target['intended_label'] for i in unseen)/len(unseen)
        full_accuracy = sum(p == y for p, y in zip(predictions, label_list))/len(label_list)
        return dict(clean_test_accuracy=correct / total, full_test_accuracy=full_accuracy,
                    clean_test_loss=loss_sum / total, targets=targets)

    def fit(self, name, method, seed, weight, case=None, delta=None, max_batches=None):
        result_path = self.out / 'runs' / (name + '.json')
        model_path = self.out / 'models' / (name + '.pt')
        if result_path.exists() and model_path.exists():
            return json.loads(result_path.read_text()), load_state(model_path)
        seed_all(seed)
        model = get_model(self.model_name, self.args.dataset).to(self.device, memory_format=torch.channels_last)
        defs = training_strategy(self.model_name, self.args)
        optimizer, scheduler = get_optimizers(model, self.args, defs)
        shuffle = torch.Generator(device=self.device).manual_seed(seed + 100)
        aug_rng = torch.Generator(device=self.device).manual_seed(seed + 200)
        defense_rng = torch.Generator(device=self.device).manual_seed(seed + 300)
        images = self.images
        if delta is not None:
            images = images.clone(memory_format=torch.preserve_format)
            ids = torch.tensor(case['poison_ids'], device=self.device)
            images[ids] += delta.to(self.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        history = []
        for epoch in range(self.config.epochs if max_batches is None else 1):
            model.train()
            order = torch.randperm(len(self.labels), device=self.device, generator=shuffle)
            sums = torch.zeros(3, device=self.device)
            batches = 0
            epoch_started = time.perf_counter()
            for start in range(0, len(order), 128):
                idx = order[start:start+128]
                inputs, labels = images[idx], self.labels[idx]
                randgen = torch.rand(len(idx), 4, device=self.device, generator=aug_rng)
                inputs = self.kettle.augment(inputs, randgen=randgen)
                optimizer.zero_grad(set_to_none=True)
                with self.autocast():
                    outputs, loss, metrics = paired_defense_loss(
                        model, inputs, labels, criterion, method, self.config.radius, weight,
                        self.dm, self.ds, defense_rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss in ' + name)
                loss.backward()
                optimizer.step()
                sums += torch.stack((loss.detach(), metrics['classification'], metrics['regularizer']))
                batches += 1
                if max_batches is not None and batches >= max_batches:
                    break
            scheduler.step()
            torch.cuda.synchronize()
            epoch_stats = dict(epoch=epoch+1, seconds=time.perf_counter()-epoch_started,
                               loss=float(sums[0]/batches), classification=float(sums[1]/batches),
                               regularizer=float(sums[2]/batches), lr=optimizer.param_groups[0]['lr'])
            history.append(epoch_stats)
            write_json(self.out / 'progress.json', dict(run=name, history=history))
            print(json.dumps(dict(event='epoch', run=name, **epoch_stats)), flush=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter()-started
        peak_bytes = torch.cuda.max_memory_allocated()
        metrics = self.evaluate(model)
        state = {key: value.cpu() for key, value in model.state_dict().items()}
        result = dict(name=name, method=method, seed=seed, weight=weight,
                      case_key=None if case is None else case['key'],
                      poisoned=delta is not None, train_seconds=train_seconds,
                      peak_gpu_bytes=peak_bytes, history=history, **metrics)
        if case is not None:
            result['attack_success'] = next(t['attack_success'] for t in metrics['targets']
                                            if t['case_key'] == case['key'])
        if max_batches is None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, model_path)
            write_json(result_path, result)
        del model, optimizer, images
        torch.cuda.empty_cache()
        print(json.dumps(dict(event='run_complete', name=name, clean_accuracy=metrics['clean_test_accuracy'],
                              seconds=train_seconds, peak_gpu_bytes=peak_bytes)), flush=True)
        return result, state

    def calibrate(self, state):
        path = self.out / 'weights.json'
        if path.exists():
            return json.loads(path.read_text())
        model = get_model(self.model_name, self.args.dataset).to(self.device, memory_format=torch.channels_last)
        model.load_state_dict(state)
        model.eval()
        weights = {'standard': 0., 'random-augmentation': 0.}
        calibration = {}
        with torch.no_grad():
            for method in ('prediction-consistency', 'gradient-consistency'):
                generator = torch.Generator(device=self.device).manual_seed(97531)
                ce, regularizer = 0., 0.
                for start in range(0, 1024, 128):
                    with self.autocast():
                        _, _, values = paired_defense_loss(model, self.images[start:start+128],
                            self.labels[start:start+128], criterion, method, self.config.radius, 1.,
                            self.dm, self.ds, generator)
                    ce += float(values['classification'])
                    regularizer += float(values['regularizer'])
                if regularizer <= 1e-12:
                    raise RuntimeError('Cannot calibrate a zero regularizer; choose a positive radius.')
                weights[method] = self.config.calibration_ratio * ce / regularizer
                calibration[method] = dict(mean_ce=ce/8, mean_regularizer=regularizer/8,
                                          weight=weights[method])
        write_json(self.out / 'calibration.json', dict(
            rule='lambda * mean_regularizer = calibration_ratio * mean_CE',
            data='first 1024 clean TRAINING examples, independent clean surrogate; no test/target tuning',
            calibration_ratio=self.config.calibration_ratio, values=calibration))
        write_json(path, weights)
        del model
        torch.cuda.empty_cache()
        return weights

    def brew(self, case, state):
        path = self.out / 'poisons' / (str(case['key']) + '.pt')
        info_path = path.with_suffix('.json')
        if path.exists() and info_path.exists():
            return load_state(path)['delta']
        seed_all(case['key'])
        self.args.poisonkey = str(case['key'])
        self.kettle.validset = self.full_validset
        self.kettle.random_construction()
        assert self.kettle.poison_ids.tolist() == case['poison_ids']
        assert int(self.kettle.target_ids[0]) == case['target_id']
        ids = self.kettle.poison_ids.to(self.device)
        dataset = torch.utils.data.TensorDataset(self.images[ids], self.labels[ids], self.kettle.poison_ids)
        self.kettle.poisonloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.pbatch,
                                                              num_workers=0, shuffle=False)
        victim = forest.Victim(self.args, setup=self.setup)
        victim.model.load_state_dict(state)
        witch = forest.Witch(self.args, setup=self.setup)
        started = time.perf_counter()
        delta = witch.brew(victim, self.kettle).cpu()
        torch.cuda.synchronize()
        pixel_delta = delta * self.ds.cpu()
        if float(pixel_delta.abs().max()) > self.config.eps/255 + 1e-6:
            raise RuntimeError('Poison bound violated.')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(delta=delta, poison_ids=self.kettle.poison_ids, target_id=case['target_id']), path)
        write_json(info_path, dict(case_key=case['key'], seconds=time.perf_counter()-started,
            matching_loss=witch.stat_optimal_loss, count=len(delta), pixel_linf=float(pixel_delta.abs().max()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del victim, witch
        torch.cuda.empty_cache()
        return delta

    def summarize(self):
        records = [json.loads(path.read_text()) for path in sorted((self.out/'runs').glob('*.json'))]
        expected_clean = len(self.config.seeds) if self.config.clean_controls else 0
        summary = []
        for method in METHODS:
            clean = [r for r in records if r['method'] == method and not r['poisoned'] and r['name'] != 'surrogate']
            poisoned = [r for r in records if r['method'] == method and r['poisoned']]
            if not poisoned:
                continue
            clean_acc = np.mean([r['clean_test_accuracy'] for r in clean]) if clean else None
            poison_acc = float(np.mean([r['clean_test_accuracy'] for r in poisoned]))
            per_case = [np.mean([r['attack_success'] for r in poisoned if r['case_key'] == key])
                        for key in self.config.cases if any(r['case_key'] == key for r in poisoned)]
            summary.append(dict(method=method, victim_runs=len(poisoned),
                clean_control_runs=len(clean), successes=sum(r['attack_success'] for r in poisoned),
                attack_success_rate=float(np.mean(per_case)),
                case_mean_standard_error=float(np.std(per_case, ddof=1)/np.sqrt(len(per_case))) if len(per_case)>1 else None,
                clean_control_accuracy=None if clean_acc is None else float(clean_acc),
                poisoned_training_accuracy=poison_acc,
                poisoning_accuracy_change_pp=None if clean_acc is None else 100*(poison_acc-float(clean_acc)),
                mean_train_seconds=float(np.mean([r['train_seconds'] for r in poisoned])),
                max_peak_gpu_gib=max(r['peak_gpu_bytes'] for r in poisoned)/1024**3))
        result = dict(evaluation=self.manifest['evaluation'], summary=summary,
                      expected_poisoned_runs_per_method=len(self.config.cases)*len(self.config.seeds),
                      expected_clean_runs_per_method=expected_clean,
                      complete=all(sum(r['method']==m and r['poisoned'] for r in records)
                                   ==len(self.config.cases)*len(self.config.seeds)
                                   and sum(r['method']==m and not r['poisoned'] and r['name']!='surrogate'
                                           for r in records)==expected_clean for m in METHODS),
                      records=records)
        write_json(self.out/'results.json', result)
        return result

    def run(self):
        if self.config.benchmark:
            results = [self.fit('benchmark-'+m, m, self.config.seeds[0], .01 if m=='gradient-consistency' else 1.,
                                max_batches=self.config.benchmark)[0] for m in METHODS]
            write_json(self.out/'benchmark.json', results)
            return
        _, state = self.fit('surrogate', 'standard', self.config.surrogate_seed, 0.)
        weights = self.calibrate(state)
        if self.config.clean_controls:
            for seed in self.config.seeds:
                for method in METHODS:
                    self.fit('clean-{}-{}'.format(seed, method), method, seed, weights[method])
        for case in self.cases:
            delta = self.brew(case, state)
            for seed in self.config.seeds:
                for method in METHODS:
                    self.fit('poison-{}-{}-{}'.format(case['key'], seed, method), method, seed,
                             weights[method], case, delta)
                    self.summarize()
        result = self.summarize()
        print(json.dumps(dict(event='comparison_complete', summary=result['summary'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/defense-comparison')
    parser.add_argument('--data', default='~/data')
    parser.add_argument('--dataset', choices=['CIFAR10', 'MNIST'], default='CIFAR10')
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--cases', nargs='+', type=int, default=[2000000000])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1729])
    parser.add_argument('--clean-controls', action='store_true', help='Additionally train each method on clean data.')
    parser.add_argument('--surrogate-seed', type=int, default=314159)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--attack-iterations', type=int, default=250)
    parser.add_argument('--restarts', type=int, default=8)
    parser.add_argument('--poison-batch', type=int, default=512)
    parser.add_argument('--eps', type=float, default=16.)
    parser.add_argument('--budget', type=float, default=.01)
    parser.add_argument('--radius', type=float, default=16.)
    parser.add_argument('--calibration-ratio', type=float, default=.1)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--benchmark', type=int, default=0, help='Only time this many batches per method.')
    config = parser.parse_args()
    if config.epochs < 1 or config.attack_iterations < 5 or config.restarts < 1:
        parser.error('Need positive epochs/restarts and at least 5 attack iterations.')
    if len(set(config.cases)) != len(config.cases) or len(set(config.seeds)) != len(config.seeds):
        parser.error('Cases and seeds must be unique.')
    if config.surrogate_seed in config.seeds:
        parser.error('Victim initialization must be independent of the surrogate.')
    if config.radius <= 0 or config.calibration_ratio <= 0 or config.eps <= 0 or not 0 < config.budget <= .1:
        parser.error('Need positive radii/ratio and a poison budget in (0, .1].')
    Comparison(config).run()


if __name__ == '__main__':
    main()
