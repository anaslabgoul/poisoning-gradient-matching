"""Locked 10-target x 10-seed study: a fresh poison and undefended victim per cell."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torchvision

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, load_state, write_json
from experiments.overnight_mnist import wilson
from experiments.search_efficient_conditions import configuration
from forest.victims.models import get_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def define_grid(train_labels,test_labels,class_names,excluded=()):
    """No predictions or attack outcomes enter target or poison selection."""
    train_labels=np.asarray(train_labels)
    test_labels=np.asarray(test_labels)
    target_rng=np.random.default_rng(20261111)
    intended_rng=np.random.default_rng(20261112)
    seeds=list(range(11001,11011))
    targets=[]
    runs=[]
    poison_rows=[]
    excluded=set(excluded)
    for label in range(10):
        candidates=[int(i) for i in np.flatnonzero(test_labels==label) if int(i) not in excluded]
        target_id=int(target_rng.choice(candidates))
        intended=int(intended_rng.choice([i for i in range(10) if i!=label]))
        targets.append(dict(target_index=label,target_id=target_id,true_label=label,
            true_class=class_names[label],intended_label=intended,intended_class=class_names[intended]))
        available=np.flatnonzero(train_labels==intended)
        for seed_index,seed in enumerate(seeds):
            run_index=label*10+seed_index
            ids=np.random.RandomState(seed+2000000).choice(available,size=400,replace=False)
            poison_rows.append(ids)
            runs.append(dict(run_index=run_index,key=4100000000+run_index,target_index=label,
                target_id=target_id,true_label=label,intended_label=intended,
                victim_seed=seed,poison_seed=seed+1000000,selection_seed=seed+2000000,
                poison_ids_sha256=index_hash(ids)))
    return targets,seeds,runs,np.asarray(poison_rows,dtype=np.int64)


def lock(args):
    data=Path(args.data).expanduser()
    dataset=torchvision.datasets.CIFAR10(str(data),train=True,download=False)
    test=torchvision.datasets.CIFAR10(str(data),train=False,download=False)
    source_root=Path(args.source_root)
    manifest=json.loads((source_root/'manifest.json').read_text())
    original_ids=manifest['training_subset']['original_indices']
    labels=np.asarray(dataset.targets)[original_ids]
    excluded=set()
    for path in args.exclude_manifests:
        prior=json.loads(Path(path).read_text())
        if prior.get('dataset',prior.get('config',{}).get('dataset'))=='CIFAR10':
            excluded.update(c['target_id'] for c in prior.get('cases',[]))
    targets,seeds,runs,poison_rows=define_grid(labels,test.targets,dataset.classes,excluded)
    inputs=Path(args.output)/'inputs'
    inputs.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(inputs/'poison-ids.npz',indices=poison_rows)
    for source_name,destination in [('models/surrogate.pt','source.pt'),
                                    ('manifest.json','source-manifest.json'),
                                    ('runs/surrogate.json','source-record.json')]:
        shutil.copyfile(source_root/source_name,inputs/destination)
    settings=dict(dataset='CIFAR10',model='ResNet20',train_size=10000,subset_seed=20261110,
        epochs=60,budget=.04,eps=32.,attack_loss='similarity',attack_iterations=250,
        restarts=1,poison_batch=512,precision='bf16',surrogate_seed=314159)
    for key in ('dataset','model','train_size','subset_seed','epochs','precision','surrogate_seed'):
        if manifest['config'][key]!=settings[key]:
            raise ValueError('Source configuration mismatch: '+key)
    plan=dict(design='10 fixed, distinct target images x 10 fixed victim seeds = 100 independently poisoned datasets and 100 undefended victim trainings.',
        source='One previously trained independent clean surrogate, seed 314159, reused without further training.',
        settings=settings,seeds=seeds,targets=targets,runs=runs,
        target_selection='One random image per class; previously used pilot targets excluded; no prediction or attack-success filtering.',
        target_selection_seed=20261111,intended_label_seed=20261112,
        excluded_prior_target_ids=sorted(excluded),training_subset_ids_sha256=index_hash(original_ids),
        files={p.name:sha256(p) for p in inputs.iterdir() if p.is_file()},
        metric='Exact intended wrong-label prediction on the fixed target after training from scratch.',
        randomness='Victim seed s; independent attack-optimization seed s+1000000 and poison-selection seed s+2000000. Reseed after surrogate construction.',
        uncertainty='Per-target Wilson intervals; overall two-way target/seed bootstrap after all 100 cells complete. Conditional on one fixed source model.',
        clean_controls='No extra per-seed clean models; clean surrogate predictions are reported separately and are not matched counterfactual controls.')
    write_json(Path(args.plan),plan)
    print(json.dumps(dict(seeds=seeds,targets=targets,planned_runs=len(runs)),indent=2))


def read_plan(args):
    plan_path=Path(args.plan)
    plan=json.loads(plan_path.read_text())
    root=Path(args.output).resolve()
    for name,expected in plan['files'].items():
        if sha256(root/'inputs'/name)!=expected:
            raise RuntimeError('Locked input hash mismatch: '+name)
    return root,plan


def cases_for(root,plan,target_index=None,prototypes=False):
    rows=np.load(root/'inputs/poison-ids.npz',allow_pickle=False)['indices']
    cases=[]
    for run in plan['runs']:
        if target_index is not None and run['target_index']!=target_index:
            continue
        if prototypes and run['victim_seed']!=plan['seeds'][0]:
            continue
        case=dict(run,poison_ids=rows[run['run_index']].tolist())
        if index_hash(case['poison_ids'])!=case['poison_ids_sha256']:
            raise RuntimeError('Locked poison IDs changed')
        cases.append(case)
    return cases


def make_experiment(args,root,plan,cases,directory):
    settings=plan['settings']
    setup=argparse.Namespace(data=str(Path(args.data).expanduser()),model=settings['model'],
        train_size=settings['train_size'],epochs=settings['epochs'],budget=settings['budget'],
        loss=settings['attack_loss'],restarts=settings['restarts'])
    config=configuration(setup,root/directory,[c['key'] for c in cases])
    config.seeds=plan['seeds']
    config.fixed_cases=cases
    experiment=Comparison(config)
    if index_hash(experiment.manifest['training_subset']['original_indices'])!=plan['training_subset_ids_sha256']:
        raise RuntimeError('Host training subset differs from locked local subset')
    return experiment


def prepare(args):
    root,plan=read_plan(args)
    experiment=make_experiment(args,root,plan,cases_for(root,plan,prototypes=True),'preparation')
    state=load_state(root/'inputs/source.pt')
    model=get_model(plan['settings']['model'],'CIFAR10').to(experiment.device,memory_format=torch.channels_last)
    model.load_state_dict(state)
    evaluation=experiment.evaluate(model)
    if evaluation['full_test_accuracy']<.75:
        raise RuntimeError('Fixed source has less than 75% clean test accuracy on this host')
    write_json(root/'source-evaluation.json',evaluation)
    write_json(root/'ready.json',dict(plan_sha256=sha256(args.plan),
        git_commit=experiment.manifest['git_commit'],source_sha256=sha256(root/'inputs/source.pt'),
        clean_source_accuracy=evaluation['full_test_accuracy'],prepared_unix=time.time()))
    print(json.dumps(dict(event='prepared',source_accuracy=evaluation['full_test_accuracy'],runs=100)),flush=True)


def run_target(args):
    root,plan=read_plan(args)
    ready=json.loads((root/'ready.json').read_text())
    if ready['plan_sha256']!=sha256(args.plan):
        raise RuntimeError('Plan changed after preparation')
    cases=cases_for(root,plan,target_index=args.target_index)
    if len(cases)!=10:
        raise ValueError('Each target requires exactly ten seeded runs')
    if all((root/'completed'/(str(c['run_index'])+'.json')).exists() for c in cases):
        return
    experiment=make_experiment(args,root,plan,cases,'targets/target-{:02d}'.format(args.target_index))
    if experiment.manifest['git_commit']!=ready['git_commit']:
        raise RuntimeError('Host code revision changed during the fixed-grid study')
    state=load_state(root/'inputs/source.pt')
    for case in cases:
        output=root/'completed'/(str(case['run_index'])+'.json')
        if output.exists():
            continue
        status=root/'status'/('target-'+str(args.target_index)+'.json')
        write_json(status,dict(phase='poison_generation',target_index=args.target_index,
                               seed=case['victim_seed'],run_index=case['run_index']))
        delta=experiment.brew(case,state)
        meta=json.loads((experiment.out/'poisons'/(str(case['key'])+'.json')).read_text())
        write_json(status,dict(phase='standard_training',target_index=args.target_index,
                               seed=case['victim_seed'],run_index=case['run_index']))
        result,_=experiment.fit('standard-seed-'+str(case['victim_seed']),'standard',
                                case['victim_seed'],0.,case,delta)
        target=next(t for t in result['targets'] if t['case_key']==case['key'])
        record=dict(run_index=case['run_index'],target_index=args.target_index,
            target_id=case['target_id'],true_label=case['true_label'],intended_label=case['intended_label'],
            victim_seed=case['victim_seed'],poison_seed=case['poison_seed'],selection_seed=case['selection_seed'],
            poison_ids_sha256=case['poison_ids_sha256'],poison_file_sha256=meta['sha256'],
            attack_success=target['attack_success'],target=target,
            clean_test_accuracy=result['full_test_accuracy'],training_seconds=result['train_seconds'],
            poison_generation_seconds=meta['seconds'],peak_gpu_bytes=max(result['peak_gpu_bytes'],meta['peak_gpu_bytes']),
            model_path=str((experiment.out/'models'/('standard-seed-'+str(case['victim_seed'])+'.pt')).resolve()))
        write_json(output,record)
        print(json.dumps(dict(event='grid_run_complete',**record)),flush=True)
    write_json(root/'status'/('target-'+str(args.target_index)+'.json'),dict(phase='complete',target_index=args.target_index,completed=10))


def summarize(args):
    root,plan=read_plan(args)
    records=[json.loads(p.read_text()) for p in (root/'completed').glob('*.json')]
    by_index={r['run_index']:r for r in records}
    if len(by_index)!=len(records):
        raise RuntimeError('Duplicate completed cells')
    expected={r['run_index']:r for r in plan['runs']}
    for index,record in by_index.items():
        for key in ('target_id','victim_seed','poison_seed','poison_ids_sha256'):
            if record[key]!=expected[index][key]:
                raise RuntimeError('Completed result does not match locked plan')
    per_target=[]
    for target in plan['targets']:
        rows=[r for r in records if r['target_index']==target['target_index']]
        successes=sum(r['attack_success'] for r in rows)
        per_target.append(dict(**target,completed=len(rows),expected=10,successes=successes,
            asr=successes/len(rows) if rows else None,
            wilson_95ci=wilson(successes,len(rows)) if rows else None,
            mean_accuracy=float(np.mean([r['clean_test_accuracy'] for r in rows])) if rows else None))
    per_seed=[]
    for seed in plan['seeds']:
        rows=[r for r in records if r['victim_seed']==seed]
        per_seed.append(dict(seed=seed,completed=len(rows),expected=10,
            successes=sum(r['attack_success'] for r in rows)))
    successes=sum(r['attack_success'] for r in records)
    complete=len(records)==100
    overall=dict(completed=len(records),expected=100,successes=successes,
        asr=successes/len(records) if records else None,
        mean_accuracy=float(np.mean([r['clean_test_accuracy'] for r in records])) if records else None)
    if complete:
        matrix=np.asarray([[by_index[t*10+s]['attack_success'] for s in range(10)] for t in range(10)],dtype=float)
        rng=np.random.default_rng(20261114)
        targets=rng.integers(0,10,size=(10000,10))
        seeds=rng.integers(0,10,size=(10000,10))
        draws=matrix[targets[:,:,None],seeds[:,None,:]].mean(axis=(1,2))
        overall['two_way_target_seed_bootstrap_95ci']=np.quantile(draws,[.025,.975]).tolist()
        overall['matrix']=matrix.astype(int).tolist()
    result=dict(complete=complete,overall=overall,per_target=per_target,per_seed=per_seed,
        definition=plan['metric'],uncertainty=plan['uncertainty'],records=sorted(records,key=lambda r:r['run_index']))
    write_json(root/'summary.json',result)
    print(json.dumps(dict(complete=complete,overall=overall,per_target=per_target),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['lock','prepare','target','summarize'])
    parser.add_argument('--plan',default='experiments/protocols/fixed-targets-10-seeds.json')
    parser.add_argument('--output',default='artifacts/fixed-targets-10x10')
    parser.add_argument('--data',default='~/data')
    parser.add_argument('--source-root',default='artifacts/efficient-search/resnet20-10k-e60-b04-cos-r4/development')
    parser.add_argument('--target-index',type=int)
    parser.add_argument('--exclude-manifests',nargs='*',default=[])
    args=parser.parse_args()
    if args.mode=='target' and args.target_index not in range(10):
        parser.error('A target index from 0 to 9 is required')
    try:
        {'lock':lock,'prepare':prepare,'target':run_target,'summarize':summarize}[args.mode](args)
    except BaseException:
        if args.mode in ('prepare','target'):
            name='prepare' if args.mode=='prepare' else 'target-'+str(args.target_index)
            write_json(Path(args.output)/'status'/(name+'.json'),dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
