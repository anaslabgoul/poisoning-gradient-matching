"""Exactly 3 matched CIFAR10/ResNet18 trials x (baseline + 4 fixed lambdas).

Supports scheduler time slices with exact optimizer/minibatch/RNG restoration.
Exit 75 means checkpointed work remains; 0 means all 15 conditions were attempted.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, load_state, seed_all, write_json, target_confidence
from forest.victims.bilevel import fixed_statistics
from forest.victims.bilevel import bilevel_backward
from forest.victims.models import get_model
from forest.victims.optimization_strategy import training_strategy
from forest.victims.training import get_optimizers


class SliceExpired(Exception):
    pass


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng(generators):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), dedicated=[g.get_state() for g in generators])


def restore_rng(state, generators):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    for generator, saved in zip(generators, state['dedicated']):
        generator.set_state(saved.cpu())


def optimizer_observation(model, optimizer, before):
    """Read-only post-update diagnostics. FP64 reductions avoid norm overflow."""
    def norm(values):
        return float(sum(v.detach().double().square().sum() for v in values).sqrt())
    layers, batchnorm = {}, {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            layer = name.rsplit('.', 1)[0]
            value = parameter.grad.detach().double().square().sum()
            layers[layer] = layers.get(layer, 0.) + value
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm[name] = dict(minimum_running_variance=float(module.running_var.min()),
                                   maximum_running_variance=float(module.running_var.max()),
                                   maximum_running_mean_abs=float(module.running_mean.abs().max()))
    momentum = [state['momentum_buffer'] for state in optimizer.state.values()
                if 'momentum_buffer' in state]
    return dict(parameter_norm=norm(model.parameters()),
                parameter_update_norm=norm([p.detach().double()-old.double()
                                           for p, old in zip(model.parameters(), before)]),
                momentum_norm=norm(momentum) if momentum else 0.,
                total_gradient_norm_by_layer={k: float(v.sqrt()) for k, v in layers.items()},
                batchnorm=batchnorm)


class BilevelTrials(Comparison):
    def __init__(self, config, protocol, slice_seconds):
        self.started = time.monotonic()
        self.deadline = self.started + slice_seconds
        self.stop_requested = False
        super().__init__(config)
        if len({case['target_id'] for case in self.cases}) != protocol.get('trial_count', 3):
            raise ValueError('Declared case keys must produce the protocol number of distinct targets.')
        self.protocol = protocol
        self.manifest['evaluation'] = protocol.get('evaluation', 'Three matched fixed standard-attack poison trials; full-network per-example min-max defense; not adaptive poisoning.')
        self.manifest['gradient_penalty'] = protocol['defense']['norm']
        self.manifest['batchnorm'] = protocol['defense']['batchnorm']
        self.manifest['protocol'] = protocol
        write_json(self.out/'manifest.json', self.manifest)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: setattr(self, 'stop_requested', True))
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, lambda *_: setattr(self, 'stop_requested', True))

    def out_of_time(self, margin=180):
        return self.stop_requested or time.monotonic() > self.deadline-margin

    def poison_path(self, case):
        return self.out/'poisons'/(str(case['key'])+'.pt')

    def observe_targets(self, model):
        """Observe confidence without updating BN buffers or consuming training RNG."""
        ids = [case['target_id'] for case in self.cases]
        with fixed_statistics(model), torch.no_grad(), self.autocast():
            logits = model(self.test_images[ids]).float()
        rows = []
        for case, values in zip(self.cases, logits):
            row = dict(case_key=case['key'], target_id=case['target_id'],
                       true_label=case['true_label'], intended_label=case['intended_label'])
            if torch.isfinite(values).all():
                row.update(confidence_status='finite',
                           **target_confidence(values, case['true_label'], case['intended_label']))
            else:
                row.update(confidence_status='nonfinite_logits', true_probability=None,
                           intended_probability=None, true_log_probability=None,
                           intended_log_probability=None, true_minus_intended_log_probability=None)
            rows.append(row)
        return rows

    def train_condition(self, name, seed, weight=None, case=None, delta=None):
        result_path = self.out/'runs'/(name+'.json')
        model_path = self.out/'models'/(name+'.pt')
        checkpoint_path = self.out/'checkpoints'/(name+'.pt')
        if result_path.exists():
            return json.loads(result_path.read_text())
        if self.out_of_time():
            raise SliceExpired()
        seed_all(seed)
        model = get_model(self.model_name, self.args.dataset).to(self.device)
        if getattr(self.config, 'channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        optimizer, scheduler = get_optimizers(model, self.args, training_strategy(self.model_name, self.args))
        generators = [torch.Generator(device=self.device).manual_seed(seed+offset) for offset in (100, 200, 300)]
        images = self.images
        if delta is not None:
            images = images.clone()
            ids = torch.tensor(case['poison_ids'], device=self.device)
            if len(delta) != len(ids) or not torch.isfinite(delta).all():
                raise ValueError('Saved poison shape or values are invalid.')
            if float((delta*self.ds.cpu()).abs().max()) > self.config.eps/255+1e-6:
                raise ValueError('Poison exceeds its declared pixel bound.')
            images[ids] += delta.to(self.device)
        epoch, offset, order = 0, 0, None
        history, totals, batches = [], torch.zeros(5, device=self.device), 0
        elapsed, peak = 0., 0
        last_finite_metrics = None
        last_target_confidences = None
        last_optimizer_observation = None
        if checkpoint_path.exists():
            # This pickle is created solely by this runner in our output directory;
            # Python/NumPy RNG states require weights_only=False. Never load uploads.
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            epoch, offset, order = saved['epoch'], saved['offset'], saved['order']
            history, totals, batches = saved['history'], saved['totals'], saved['batches']
            elapsed, peak = saved['elapsed'], saved['peak']
            restore_rng(saved['rng'], generators)
            del saved
        torch.cuda.reset_peak_memory_stats()
        resumed = time.monotonic()
        last_saved = resumed

        def checkpoint():
            save_checkpoint(checkpoint_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, offset=offset, order=order, history=history,
                totals=totals, batches=batches, elapsed=elapsed+time.monotonic()-resumed,
                peak=max(peak, torch.cuda.max_memory_allocated()), rng=capture_rng(generators)))

        print(json.dumps(dict(event='condition_started', name=name, epoch=epoch, offset=offset,
                              lambda_weight=weight, parameter_count=sum(p.numel() for p in model.parameters()))), flush=True)
        if getattr(self.config, 'log_target_confidences', False):
            last_target_confidences = self.observe_targets(model)
            print(json.dumps(dict(event='target_confidence', condition=name, epoch=epoch,
                                  offset=offset, targets=last_target_confidences)), flush=True)
        try:
            while epoch < self.config.epochs:
                model.train()
                if order is None:
                    order = torch.randperm(len(self.labels), device=self.device, generator=generators[0])
                while offset < len(order):
                    if self.out_of_time():
                        checkpoint()
                        raise SliceExpired()
                    idx = order[offset:offset+self.config.batch_size]
                    x, y = images[idx], self.labels[idx]
                    randgen = torch.rand(len(idx), 4, device=self.device, generator=generators[1])
                    x = self.kettle.augment(x, randgen=randgen)
                    optimizer.zero_grad(set_to_none=True)
                    diagnose_update = getattr(self.config, 'update_diagnostics', False) and (
                        (epoch==0 and batches<32) or time.monotonic()-last_saved>self.config.checkpoint_seconds)
                    if weight is None:
                        loss = F.cross_entropy(model(x), y)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite classification loss.')
                        loss.backward()
                        metrics = dict(loss=loss.detach(), classification=loss.detach(),
                            regularizer=loss.detach()*0, inner_initial=loss.detach()*0, inner_maximized=loss.detach()*0)
                    else:
                        diagnose_gradients = getattr(self.config, 'gradient_diagnostics', False) and (
                            (epoch==0 and batches<32) or time.monotonic()-last_saved>self.config.checkpoint_seconds)
                        _, metrics = bilevel_backward(model, x, y, weight,
                            radius=self.config.radius, steps=self.config.inner_steps,
                            chunk_size=self.config.gradient_chunk_size, dm=self.dm, ds=self.ds,
                            generator=generators[2], step_size=self.config.inner_step_size,
                            classification_bf16=getattr(self.config, 'precision', 'fp32')=='bf16',
                            regularizer_samples=getattr(self.config, 'regularizer_samples', None),
                            gradient_diagnostics=diagnose_gradients)
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('Nonfinite outer parameter gradient.')
                    before = [p.detach().clone() for p in model.parameters()] if diagnose_update else None
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise FloatingPointError('Nonfinite updated parameter.')
                    if getattr(self.config, 'validate_buffers', False) and not all(
                            torch.isfinite(buffer).all() for buffer in model.buffers()):
                        raise FloatingPointError('Nonfinite normalization buffer.')
                    if diagnose_update:
                        last_optimizer_observation = optimizer_observation(model, optimizer, before)
                        del before
                    totals += torch.stack([metrics[k] for k in ('loss','classification','regularizer','inner_initial','inner_maximized')])*len(idx)
                    offset += len(idx)
                    batches += 1
                    progress = dict(phase='training', condition=name, epoch=epoch+1, epochs=self.config.epochs,
                        processed=offset, train_images=len(order), lambda_weight=weight,
                        metrics={k: float(v) for k, v in metrics.items()}, pid=os.getpid())
                    if diagnose_update:
                        progress['optimizer_diagnostics'] = last_optimizer_observation
                    if getattr(self.config, 'log_target_confidences', False) and (
                            (epoch==0 and batches<=32) or diagnose_update):
                        last_target_confidences = self.observe_targets(model)
                        progress['target_confidences'] = last_target_confidences
                    last_finite_metrics = progress['metrics']
                    write_json(self.out/'status.json', progress)
                    if getattr(self.config, 'gradient_diagnostics', False) and epoch==0 and batches<=32:
                        print(json.dumps(dict(event='early_batch', batch=batches, **progress)),flush=True)
                        if batches==32:
                            print(json.dumps(dict(event='early_stability_passed',condition=name,batches=32,
                                interpretation='Continue this same training; early finiteness does not establish full stability.')),flush=True)
                    if time.monotonic()-last_saved > self.config.checkpoint_seconds:
                        checkpoint()
                        last_saved = time.monotonic()
                        print(json.dumps(dict(event='checkpoint', **progress)), flush=True)
                scheduler.step()
                epoch += 1
                row = dict(epoch=epoch, **{k: float(v/len(order)) for k,v in zip(
                    ('loss','classification','regularizer','inner_initial','inner_maximized'), totals)},
                    lr=optimizer.param_groups[0]['lr'])
                if getattr(self.config, 'log_target_confidences', False):
                    last_target_confidences = self.observe_targets(model)
                    row['target_confidences'] = last_target_confidences
                history.append(row)
                print(json.dumps(dict(event='epoch', condition=name, **row)), flush=True)
                order, offset, totals, batches = None, 0, torch.zeros(5, device=self.device), 0
                checkpoint()
                last_saved = time.monotonic()
            metrics = self.evaluate(model)
            result = dict(name=name, status='complete', method='standard' if weight is None else 'full-network-bilevel',
                seed=seed, lambda_weight=weight, case_key=None if case is None else case['key'],
                target_id=None if case is None else case['target_id'], poisoned=delta is not None,
                train_seconds=elapsed+time.monotonic()-resumed,
                peak_gpu_bytes=max(peak, torch.cuda.max_memory_allocated()), history=history, **metrics)
            if case is not None:
                target = next(t for t in metrics['targets'] if t['case_key']==case['key'])
                result.update(attack_success=target['attack_success'], target=target,
                    poison_sha256=hashlib.sha256(self.poison_path(case).read_bytes()).hexdigest())
            save_checkpoint(model_path, {k:v.cpu() for k,v in model.state_dict().items()})
            write_json(result_path, result)
            # This file belongs only to this named condition and is superseded by
            # the final checkpoint and result, which are both durable first.
            checkpoint_path.unlink(missing_ok=True)
            print(json.dumps(dict(event='condition_complete', name=name,
                attack_success=result.get('attack_success'), accuracy=metrics['clean_test_accuracy'])), flush=True)
            return result
        except FloatingPointError as error:
            if case is None:
                raise
            result = dict(name=name, status='numerical_failure', seed=seed, lambda_weight=weight,
                case_key=case['key'], target_id=case['target_id'], epoch=epoch+1, offset=offset,
                error=str(error), history=history,
                last_finite_metrics=last_finite_metrics,
                interpretation='Failed optimization; never count this as a prevented attack or silently alter lambda.')
            if getattr(self.config, 'log_target_confidences', False):
                result['last_observed_target_confidences'] = last_target_confidences
                result['final_confidence_available'] = False
            if getattr(self.config, 'save_failure_checkpoint', False):
                failure_path = self.out/'failures'/(name+'.pt')
                save_checkpoint(failure_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, offset=offset, order=order,
                    rng=capture_rng(generators), last_finite_metrics=last_finite_metrics,
                    last_optimizer_observation=last_optimizer_observation,
                    last_target_confidences=last_target_confidences, error=str(error)))
                result['failure_checkpoint'] = str(failure_path)
            write_json(result_path, result)
            print(json.dumps(dict(event='condition_failed', **result)), flush=True)
            return result
        finally:
            del model, optimizer, images
            torch.cuda.empty_cache()

    def summarize_trials(self):
        rows = [json.loads(p.read_text()) for p in sorted((self.out/'runs').glob('trial-*.json'))]
        by_name = {r['name']:r for r in rows}
        summary = []
        for label, weight in [('standard', None)]+[('lambda-'+str(w),w) for w in self.protocol['lambdas']]:
            selected = [r for r in rows if r['lambda_weight']==weight]
            completed = [r for r in selected if r['status']=='complete']
            pairs = []
            for row in completed:
                base = by_name.get('trial-{}-standard'.format(self.config.cases.index(row['case_key'])))
                if base and base['status']=='complete':
                    pairs.append(dict(case_key=row['case_key'], prevented=base['attack_success']-row['attack_success'],
                                      accuracy_change_pp=100*(row['clean_test_accuracy']-base['clean_test_accuracy'])))
            summary.append(dict(condition=label, lambda_weight=weight, attempted=len(selected), completed=len(completed),
                failed=len(selected)-len(completed), successes=sum(r['attack_success'] for r in completed),
                targeted_asr=None if not completed else float(np.mean([r['attack_success'] for r in completed])),
                accuracy=None if not completed else float(np.mean([r['clean_test_accuracy'] for r in completed])),
                paired=pairs))
        expected = self.protocol.get('victim_models', 15)
        result = dict(attempted=len(rows), expected=expected, all_attempted=len(rows)==expected,
            all_succeeded=len(rows)==expected and all(r['status']=='complete' for r in rows), summary=summary, records=rows,
            interpretation=self.protocol['inference'], uncertainty=self.protocol.get('uncertainty', 'Only three independent target/seed pairs; raw paired outcomes and descriptive means, no significance claim. Failed conditions are excluded from rates and explicitly counted.'))
        write_json(self.out/'results.json', result)
        return result

    def run_trials(self):
        self.summarize_trials()
        source = self.train_condition('surrogate', self.config.surrogate_seed)
        if source['clean_test_accuracy'] < .85:
            raise RuntimeError('Clean surrogate accuracy below 85%; inspect before brewing misleading attacks.')
        state = load_state(self.out/'models/surrogate.pt')
        for index, case in enumerate(self.cases):
            poison_path = self.out/'poisons'/(str(case['key'])+'.pt')
            if not poison_path.exists() and self.out_of_time(margin=1800):
                raise SliceExpired()
            write_json(self.out/'status.json', dict(phase='brewing', trial=index, case_key=case['key']))
            delta = self.brew(case, state)
            if int(case['target_id']) != int(load_state(poison_path)['target_id']):
                raise RuntimeError('Poison target mismatch.')
            conditions = [('standard', None)]+[('lambda-'+str(w), w) for w in self.protocol['lambdas']]
            for label, weight in conditions:
                self.train_condition('trial-{}-{}'.format(index,label), self.config.seeds[index], weight, case, delta)
                self.summarize_trials()
        final = self.summarize_trials()
        write_json(self.out/'status.json', dict(phase='complete' if final['all_succeeded'] else 'complete_with_failures',
            attempted=final['attempted'], expected=self.protocol.get('victim_models', 15)))


def make_config(args, protocol):
    attack, defense = protocol['attack'], protocol['defense']
    return argparse.Namespace(output=str(Path(args.output).resolve()), data=args.data,
        dataset=protocol['dataset'], model=protocol['model'], train_size=None,
        cases=protocol['case_keys'], seeds=protocol['victim_seeds'], surrogate_seed=protocol['surrogate_seed'],
        epochs=protocol['epochs'], batch_size=protocol['batch_size'],
        attack_iterations=attack['iterations'], restarts=attack['restarts'], poison_batch=attack['poison_batch'],
        attack_loss=attack['loss'], eps=attack['epsilon_pixel_units'], budget=attack['budget'],
        radius=defense['radius_pixel_units'], inner_steps=defense['inner_steps'],
        inner_step_size=defense['inner_step_pixel_units'], gradient_chunk_size=defense['gradient_chunk_size'],
        protocol=copy.deepcopy(protocol), precision='fp32', threads=4, checkpoint_seconds=120,
        calibration_ratio=.1, clean_controls=False, benchmark=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='experiments/protocols/bilevel-resnet18-three-trials.json')
    parser.add_argument('--output', default='artifacts/bilevel-resnet18')
    parser.add_argument('--data', default='~/data')
    parser.add_argument('--slice-seconds', type=float, default=6000.)
    args = parser.parse_args()
    root = Path(args.output)
    protocol = json.loads(Path(args.protocol).read_text())
    if (protocol['dataset']!='CIFAR10' or protocol['model']!='ResNet18' or
        len(protocol['case_keys'])!=3 or len(set(protocol['case_keys']))!=3 or
        len(protocol['victim_seeds'])!=3 or set(protocol['lambdas'])!={1.,.1,.001,.0001}):
        raise ValueError('Protocol must match the explicitly requested 3 trials and 4 lambdas.')
    try:
        experiment = BilevelTrials(make_config(args, protocol), protocol, args.slice_seconds)
        experiment.run_trials()
    except SliceExpired:
        write_json(root/'status.json', dict(phase='checkpointed', reason='Scheduler slice complete; continue automatically.'))
        return 75
    except BaseException:
        write_json(root/'status.json', dict(phase='failed', traceback=traceback.format_exc()))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
