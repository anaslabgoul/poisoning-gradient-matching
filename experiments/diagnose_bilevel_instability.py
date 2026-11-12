"""Bounded replay of the failed overnight prefix; never retrain full victims.

Read the immutable original manifest and poison. Each diagnostic intervention
starts from the original seed and runs at most --steps minibatches. Results
are written separately; the original study artifacts are never modified.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, load_state, seed_all, write_json
from forest.victims import bilevel
from forest.victims.models import get_model
from forest.victims.optimization_strategy import training_strategy
from forest.victims.training import get_optimizers


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def norm(values):
    return float(sum(v.detach().double().square().sum() for v in values).sqrt())


def tensor_summary(value):
    value = value.detach()
    return dict(finite=bool(torch.isfinite(value).all()), absmax=float(value.abs().max()),
                minimum=float(value.min()), maximum=float(value.max()))


def parameter_summary(model):
    return dict(norm=norm(model.parameters()), absmax=max(float(p.detach().abs().max()) for p in model.parameters()),
                finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))


def bn_summary(model):
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
    return dict(minimum_running_variance=min(float(m.running_var.min()) for _,m in layers),
                maximum_running_variance=max(float(m.running_var.max()) for _,m in layers),
                maximum_running_mean_abs=max(float(m.running_mean.abs().max()) for _,m in layers),
                finite=all(bool(torch.isfinite(m.running_var).all() and torch.isfinite(m.running_mean).all()) for _,m in layers),
                max_affine_abs=max(float(m.weight.detach().abs().max()) for _,m in layers))


@contextmanager
def batch_statistics(model, statistics, enabled):
    saved = []
    try:
        if enabled:
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        saved.append((module, module.running_mean.clone(), module.running_var.clone()))
                        mean, variance = statistics[name]
                        module.running_mean.copy_(mean)
                        module.running_var.copy_(variance)
        yield
    finally:
        with torch.no_grad():
            for module, mean, variance in saved:
                module.running_mean.copy_(mean)
                module.running_var.copy_(variance)


def replay(experiment, source, original, variant, steps, output):
    config = experiment.config
    seed = original['config']['seeds'][0]
    weight = 1e-6 if variant=='small-lambda' else .001
    inner_steps = 0 if variant=='random' else 3
    seed_all(seed)
    model = get_model('ResNet18', 'CIFAR10').to(experiment.device)
    optimizer, _ = get_optimizers(model, experiment.args, training_strategy('ResNet18', experiment.args))
    if variant=='small-lr':
        for group in optimizer.param_groups:
            group['lr'] = .001
    generators = [torch.Generator(device=experiment.device).manual_seed(seed+offset) for offset in (100,200,300)]
    images = experiment.images.clone()
    case = original['cases'][0]
    poison_path = source/'poisons'/(str(case['key'])+'.pt')
    poison = load_state(poison_path)
    ids = torch.tensor(case['poison_ids'], device=experiment.device)
    if ids.tolist()!=poison['poison_ids'].tolist() or int(poison['target_id'])!=case['target_id']:
        raise RuntimeError('Original poison row/target identity mismatch.')
    if hashlib.sha256(poison_path.read_bytes()).hexdigest()!=json.loads(poison_path.with_suffix('.json').read_text())['sha256']:
        raise RuntimeError('Original poison hash mismatch.')
    if variant!='clean':
        images[ids] += poison['delta'].to(experiment.device)
    order = torch.randperm(len(images), device=experiment.device, generator=generators[0])
    mask = torch.zeros(len(images), dtype=torch.bool, device=experiment.device)
    mask[ids] = True
    result = dict(variant=variant, lambda_weight=0. if variant=='ce-only' else weight,
        learning_rate=optimizer.param_groups[0]['lr'], inner_steps=inner_steps,
        altered_bn=variant=='batch-bn', original_execution_commit=original['git_commit'],
        source_poison_sha256=hashlib.sha256(poison_path.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cudnn_benchmark=torch.backends.cudnn.benchmark,
        description='Diagnostic prefix only; no attack-success or generalization conclusions.',
        steps=[], outcome='running')
    trace_path = output/variant/'trace.json'
    started = time.monotonic()
    try:
        for index in range(steps):
            model.train()
            rows = order[index*128:(index+1)*128]
            x, y = images[rows], experiment.labels[rows]
            randgen = torch.rand(len(rows),4,device=experiment.device,generator=generators[1])
            x = experiment.kettle.augment(x,randgen=randgen)
            record = dict(step=index+1, batch_rows=rows.cpu().tolist(), poisoned_rows=int(mask[rows].sum()),
                          before_parameters=parameter_summary(model), before_bn=bn_summary(model))
            result['steps'].append(record)
            optimizer.zero_grad(set_to_none=True)
            if variant=='production':
                # Cross-check the instrumented arithmetic against the untouched
                # production entry point, without any activation/gradient hooks.
                _, metrics = bilevel.bilevel_backward(model,x,y,.001,radius=16.,steps=3,
                    chunk_size=8,dm=experiment.dm,ds=experiment.ds,generator=generators[2],
                    step_size=config.inner_step_size,regularizer_samples=8)
                record.update({k:float(v) for k,v in metrics.items()})
                record['total_gradient_norm']=norm([p.grad for p in model.parameters()])
                if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite outer parameter gradient.')
                optimizer.step()
                record['after_parameters']=parameter_summary(model)
                record['after_bn']=bn_summary(model)
                write_json(trace_path,safe(result))
                print(json.dumps(safe(dict(variant=variant,step=index+1,**{k:record[k] for k in
                    ('classification','regularizer','total_gradient_norm','after_parameters','after_bn')}))),flush=True)
                if not record['after_parameters']['finite']:
                    raise FloatingPointError('Nonfinite updated parameter.')
                continue
            statistics, activations, handles = {}, {}, []
            def hook(name, module):
                def capture(layer, args, output):
                    activations[name] = tensor_summary(output)
                    if isinstance(module, torch.nn.BatchNorm2d):
                        value = args[0].detach()
                        statistics[name] = (value.mean(dim=(0,2,3)), value.var(dim=(0,2,3),unbiased=False))
                return capture
            for name, module in model.named_modules():
                if isinstance(module,(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.Linear)):
                    handles.append(module.register_forward_hook(hook(name,module)))
            try:
                logits = model(x)
            finally:
                for handle in handles:
                    handle.remove()
            loss = F.cross_entropy(logits,y)
            record.update(classification=float(loss.detach()), logits=tensor_summary(logits),
                          classification_activations=activations, after_ce_bn=bn_summary(model))
            if not torch.isfinite(loss):
                record['first_nonfinite_activation'] = next((k for k,v in activations.items() if not v['finite']),None)
                result['outcome'] = 'nonfinite_classification'
                break
            loss.backward()
            ce_gradients = [p.grad.detach().clone() for p in model.parameters()]
            record['ce_gradient_norm'] = norm(ce_gradients)
            if variant!='ce-only':
                selected = torch.randperm(len(x),device=experiment.device,generator=generators[2])[:8]
                record['selected_penalty_rows'] = rows[selected].cpu().tolist()
                record['selected_poisoned_rows'] = int(mask[rows[selected]].sum())
                a,b = x[selected],y[selected]
                with batch_statistics(model,statistics,variant=='batch-bn'):
                    inner_traces = []
                    original_difference = bilevel.squared_difference
                    def trace_difference(first,second):
                        values = original_difference(first,second)
                        inner_traces.append(values.detach().cpu().tolist())
                        return values
                    bilevel.squared_difference = trace_difference
                    try:
                        delta, diagnostics = bilevel.maximize_gradient_difference(model,a,b,
                            radius=16.,steps=inner_steps,step_size=config.inner_step_size,
                            dm=experiment.dm,ds=experiment.ds,generator=generators[2])
                        penalty = bilevel.gradient_penalties(model,a,b,delta,experiment.dm,experiment.ds).mean()
                        record.update(regularizer=float(penalty.detach()),weighted_regularizer=float(weight*penalty.detach()),
                            inner_initial=float(diagnostics['initial'].mean()),inner_maximized=float(diagnostics['maximized'].mean()),
                            inner_scores=inner_traces, pixel_delta_max=float(delta.abs().max()),
                            inner_and_outer_score_relative_difference=float((penalty.detach()-diagnostics['maximized'].mean()).abs()/(penalty.detach().abs()+1e-30)))
                        (weight*penalty).backward()
                    finally:
                        bilevel.squared_difference = original_difference
                del penalty,delta
            regularizer_gradients = [p.grad.detach()-g for p,g in zip(model.parameters(),ce_gradients)]
            record['weighted_regularizer_gradient_norm'] = norm(regularizer_gradients)
            record['total_gradient_norm'] = norm([p.grad for p in model.parameters()])
            record['regularizer_to_ce_gradient_ratio'] = record['weighted_regularizer_gradient_norm']/max(record['ce_gradient_norm'],1e-30)
            layer_norms = [(name,float(g.double().square().sum().sqrt())) for (name,_),g in zip(model.named_parameters(),regularizer_gradients)]
            record['largest_regularizer_gradient_layers'] = sorted(layer_norms,key=lambda pair:pair[1],reverse=True)[:8]
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                result['outcome'] = 'nonfinite_outer_gradient'
                break
            before = [p.detach().clone() for p in model.parameters()]
            optimizer.step()
            record['actual_parameter_update_norm'] = norm([p.detach()-q for p,q in zip(model.parameters(),before)])
            record['update_to_parameter_ratio'] = record['actual_parameter_update_norm']/record['before_parameters']['norm']
            record['after_parameters'] = parameter_summary(model)
            record['after_bn'] = bn_summary(model)
            del ce_gradients,regularizer_gradients,before
            write_json(trace_path,safe(result))
            print(json.dumps(safe(dict(variant=variant,**{k:record.get(k) for k in
                ('step','classification','regularizer','weighted_regularizer','ce_gradient_norm',
                 'weighted_regularizer_gradient_norm','regularizer_to_ce_gradient_ratio',
                 'actual_parameter_update_norm','update_to_parameter_ratio','after_parameters','after_bn')}))),flush=True)
            if not record['after_parameters']['finite']:
                result['outcome'] = 'nonfinite_updated_parameters'
                break
        else:
            result['outcome'] = 'finite_prefix'
    except (FloatingPointError,RuntimeError) as error:
        result['outcome'] = 'diagnostic_exception'
        result['error'] = str(error)
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write_json(trace_path,safe(result))
        print(json.dumps(safe(dict(event='diagnosis_complete',variant=variant,outcome=result['outcome'],
                                  steps=len(result['steps']),seconds=result['elapsed_seconds']))),flush=True)
        del model,optimizer,images
        torch.cuda.empty_cache()
    return safe(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='artifacts/bilevel-overnight')
    parser.add_argument('--output',default='artifacts/bilevel-diagnosis')
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--data',default=None)
    parser.add_argument('--no-autotune',action='store_true')
    parser.add_argument('--variants',nargs='+',choices=['original','production','random','ce-only','clean','small-lr','small-lambda','batch-bn'],
                        default=['original','random','ce-only','clean','small-lr','small-lambda','batch-bn'])
    args = parser.parse_args()
    if not 1<=args.steps<=32:
        parser.error('Diagnostic prefixes are limited to 1--32 minibatches.')
    source,output = Path(args.source).resolve(),Path(args.output).resolve()
    original = json.loads((source/'manifest.json').read_text())
    config = argparse.Namespace(**original['config'])
    config.output = str(output/'context')
    config.fixed_cases = original['cases']
    if args.data:
        config.data = str(Path(args.data).resolve())
    experiment = Comparison(config)
    if args.no_autotune:
        torch.backends.cudnn.benchmark=False
    results=[]
    for variant in args.variants:
        results.append(replay(experiment,source,original,variant,args.steps,output))
        write_json(output/'diagnosis.json',dict(source=str(source),requested_steps=args.steps,results=results))


if __name__=='__main__':
    main()
