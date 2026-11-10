"""Apply gradient consistency to exactly the saved, completed baseline cohort.

No attacks or standard models are regenerated. The stopped 100-cell grid is
never resumed. Only the 25 explicitly frozen baseline cells are evaluated.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison,load_state,write_json
from experiments.fixed_target_grid import cases_for,index_hash,read_plan,sha256
from experiments.search_efficient_conditions import configuration


def resolve_lambda(calibrated, override):
    weight = calibrated if override is None else override
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Lambda must be finite and nonnegative.')
    return float(weight)


def frozen_cohort(summary,expected=25):
    rows=sorted(summary['records'],key=lambda r:r['run_index'])
    if len(rows)!=expected or len({r['run_index'] for r in rows})!=expected:
        raise ValueError('Require exactly '+str(expected)+' unique completed baseline runs.')
    if any(r['target_id']!=r['target']['target_id'] for r in rows):
        raise ValueError('Baseline target mismatch')
    return dict(expected=expected,run_indices=[r['run_index'] for r in rows],records=rows,
        baseline_successes=sum(r['attack_success'] for r in rows),
        baseline_asr=sum(r['attack_success'] for r in rows)/expected,
        baseline_accuracy=float(np.mean([r['clean_test_accuracy'] for r in rows])),
        defense='gradient-consistency',radius=32.,calibration_ratio=.1,
        formula='CE(original) + lambda * mean per-example squared final-weight-gradient difference; one uniform random perturbation per image, not inner maximization.',
        scope='Only this stopped 25-run cohort. No poison generation, no additional standard runs, no other defenses.')


def aggregate(cohort,records):
    expected={r['run_index']:r for r in cohort['records']}
    if len({r['run_index'] for r in records})!=len(records):
        raise ValueError('Duplicate defense outcomes')
    for row in records:
        base=expected[row['run_index']]
        if any(row[k]!=base[k] for k in ('target_id','victim_seed','poison_file_sha256')):
            raise ValueError('Defended result does not match frozen baseline')
    n=len(records)
    result=dict(complete=n==cohort['expected'],completed=n,expected=cohort['expected'],
        frozen_baseline=dict(trials=cohort['expected'],successes=cohort['baseline_successes'],
            asr=cohort['baseline_asr'],mean_clean_accuracy=cohort['baseline_accuracy']),
        method=cohort['defense'],records=sorted(records,key=lambda r:r['run_index']))
    if not n:
        return result
    baseline_success=sum(expected[r['run_index']]['attack_success'] for r in records)
    defended_success=sum(r['attack_success'] for r in records)
    base_accuracy=np.asarray([expected[r['run_index']]['clean_test_accuracy'] for r in records])
    defended_accuracy=np.asarray([r['clean_test_accuracy'] for r in records])
    result['paired']=dict(trials=n,baseline_successes=baseline_success,defended_successes=defended_success,
        baseline_asr=baseline_success/n,defended_asr=defended_success/n,
        asr_reduction_pp=100*(baseline_success-defended_success)/n,
        relative_asr_reduction=(baseline_success-defended_success)/baseline_success if baseline_success else None,
        baseline_accuracy=float(base_accuracy.mean()),defended_accuracy=float(defended_accuracy.mean()),
        accuracy_change_pp=float(100*(defended_accuracy-base_accuracy).mean()),
        attacks_prevented=sum(expected[r['run_index']]['attack_success']==1 and r['attack_success']==0 for r in records),
        attacks_introduced=sum(expected[r['run_index']]['attack_success']==0 and r['attack_success']==1 for r in records))
    result['per_target']=[]
    for target_id in sorted({r['target_id'] for r in cohort['records']}):
        rows=[r for r in records if r['target_id']==target_id]
        if not rows:
            continue
        bases=[expected[r['run_index']] for r in rows]
        b=sum(r['attack_success'] for r in bases)
        d=sum(r['attack_success'] for r in rows)
        result['per_target'].append(dict(target_id=target_id,trials=len(rows),
            true_label=bases[0]['true_label'],intended_label=bases[0]['intended_label'],
            baseline_successes=b,defended_successes=d,asr_reduction_pp=100*(b-d)/len(rows),
            accuracy_change_pp=100*float(np.mean([r['clean_test_accuracy']-expected[r['run_index']]['clean_test_accuracy'] for r in rows]))))
    if result['complete']:
        seeds=sorted({r['victim_seed'] for r in records})
        counts=[]
        attack_sums=[]
        accuracy_sums=[]
        for seed in seeds:
            block=[r for r in records if r['victim_seed']==seed]
            counts.append(len(block))
            attack_sums.append(sum(expected[r['run_index']]['attack_success']-r['attack_success'] for r in block))
            accuracy_sums.append(sum(r['clean_test_accuracy']-expected[r['run_index']]['clean_test_accuracy'] for r in block))
        draws=np.random.default_rng(20261115).integers(0,len(seeds),size=(10000,len(seeds)))
        denominators=np.asarray(counts)[draws].sum(1)
        attack_draws=100*np.asarray(attack_sums)[draws].sum(1)/denominators
        accuracy_draws=100*np.asarray(accuracy_sums)[draws].sum(1)/denominators
        result['paired']['asr_reduction_seed_bootstrap_95ci_pp']=np.quantile(attack_draws,[.025,.975]).tolist()
        result['paired']['accuracy_change_seed_bootstrap_95ci_pp']=np.quantile(accuracy_draws,[.025,.975]).tolist()
        result['uncertainty']='Descriptive paired seed-block bootstrap, 10,000 draws; conditional on these three fixed target images. Not a population-wide robustness claim.'
    return result


def summarize(root,cohort):
    records=[json.loads(p.read_text()) for p in (root/'completed').glob('*.json')]
    result=aggregate(cohort,records)
    write_json(root/'comparison.json',result)
    return result


def run(args):
    cohort=json.loads(Path(args.cohort).read_text())
    if cohort['expected']!=25:
        raise ValueError('This launch is limited to the 25 reported baseline runs.')
    override=getattr(args,'lambda_weight',None)
    if override is not None:
        resolve_lambda(0.,override)
    baseline_root,plan=read_plan(argparse.Namespace(plan=args.plan,output=args.baseline))
    ready=json.loads((baseline_root/'ready.json').read_text())
    if sha256(args.plan)!=ready['plan_sha256']:
        raise RuntimeError('The original baseline plan has changed')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    selected={r['run_index']:r for r in cohort['records']}
    cases=[c for c in cases_for(baseline_root,plan) if c['run_index'] in selected]
    if len(cases)!=25:
        raise ValueError('Frozen cohort must map to exactly 25 saved poisons')
    # Verify the whole frozen cohort before starting any defense training.
    poison_paths={}
    for case in cases:
        base=selected[case['run_index']]
        if any(case[k]!=base[k] for k in ('target_id','true_label','intended_label','victim_seed','poison_seed','poison_ids_sha256')):
            raise RuntimeError('Frozen baseline differs from the original target/seed plan')
        saved=json.loads((baseline_root/'completed'/(str(case['run_index'])+'.json')).read_text())
        if any(saved[k]!=base[k] for k in ('victim_seed','target_id','attack_success','poison_file_sha256','poison_ids_sha256')):
            raise RuntimeError('Saved baseline differs from frozen cohort')
        path=baseline_root/'targets'/('target-{:02d}'.format(case['target_index']))/'poisons'/(str(case['key'])+'.pt')
        if sha256(path)!=base['poison_file_sha256']:
            raise RuntimeError('Saved poison changed: '+str(case['run_index']))
        poison_paths[case['run_index']]=path
    settings=plan['settings']
    setup=argparse.Namespace(data=str(Path(args.data).expanduser()),model=settings['model'],
        train_size=settings['train_size'],epochs=settings['epochs'],budget=settings['budget'],
        loss=settings['attack_loss'],restarts=settings['restarts'])
    config=configuration(setup,root/'models-and-runs',[c['key'] for c in cases])
    config.fixed_cases=cases
    config.seeds=plan['seeds']
    config.radius=cohort['radius']
    config.calibration_ratio=cohort['calibration_ratio']
    # Include the override in the manifest so a resume cannot mix lambda values.
    config.lambda_weight=override
    experiment=Comparison(config)
    if index_hash(experiment.manifest['training_subset']['original_indices'])!=plan['training_subset_ids_sha256']:
        raise RuntimeError('Training subset differs from baseline')
    state=load_state(baseline_root/'inputs/source.pt')
    weights=experiment.calibrate(state)
    weight=resolve_lambda(weights['gradient-consistency'],override)
    write_json(root/'protocol.json',dict(cohort=cohort,lambda_gradient=weight,
        lambda_source='clean-training calibration' if override is None else 'fixed sweep value',
        calibrated_reference_lambda=weights['gradient-consistency'],
        calibration='Reference lambda * mean penalty = 0.1 * mean CE; first 1024 clean TRAINING images, frozen clean source. Explicit lambda overrides this reference for a sweep.',
        matched='Identical saved poison tensor, target, intended label, victim initialization seed, shuffle and spatial augmentation draws.',
        evaluation='Fixed standard-attack poisons, not adapted to the defense.',
        source_checkpoint_sha256=sha256(baseline_root/'inputs/source.pt'),
        baseline_code_commit=json.loads((baseline_root/'ready.json').read_text())['git_commit'],
        defense_code_commit=experiment.manifest['git_commit']))
    started=time.time()
    for index,case in enumerate(cases):
        record_path=root/'completed'/(str(case['run_index'])+'.json')
        if record_path.exists():
            continue
        saved=load_state(poison_paths[case['run_index']])
        if saved['poison_ids'].tolist()!=case['poison_ids'] or int(saved['target_id'])!=case['target_id']:
            raise RuntimeError('Poison row/target alignment differs')
        delta=saved['delta']
        if not torch.isfinite(delta).all() or float((delta*experiment.ds.cpu()).abs().max())>32/255+1e-6:
            raise RuntimeError('Poison values are invalid')
        write_json(root/'status.json',dict(phase='training',completed=index,expected=25,
            run_index=case['run_index'],target_id=case['target_id'],seed=case['victim_seed']))
        result,_=experiment.fit('gradient-run-{:03d}'.format(case['run_index']),'gradient-consistency',
            case['victim_seed'],weight,case,delta)
        target=next(t for t in result['targets'] if t['case_key']==case['key'])
        record=dict(run_index=case['run_index'],target_id=case['target_id'],victim_seed=case['victim_seed'],
            poison_file_sha256=selected[case['run_index']]['poison_file_sha256'],
            attack_success=target['attack_success'],target=target,clean_test_accuracy=result['full_test_accuracy'],
            training_seconds=result['train_seconds'],lambda_gradient=weight)
        write_json(record_path,record)
        comparison=summarize(root,cohort)
        print(json.dumps(dict(event='defense_pair_complete',completed=comparison['completed'],paired=comparison['paired'])),flush=True)
    final=summarize(root,cohort)
    write_json(root/'status.json',dict(phase='complete',completed=final['completed'],expected=25,wall_seconds=time.time()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',default='experiments/protocols/gradient-defense-cohort-25.json')
    parser.add_argument('--plan',default='experiments/protocols/fixed-targets-10-seeds.json')
    parser.add_argument('--baseline',default='artifacts/fixed-targets-10x10')
    parser.add_argument('--output',default='artifacts/gradient-defense-25')
    parser.add_argument('--data',default='~/data')
    parser.add_argument('--lambda-weight',type=float,default=None,
                        help='Fixed nonnegative coefficient; otherwise use clean-training calibration.')
    args=parser.parse_args()
    try:
        run(args)
    except BaseException:
        write_json(Path(args.output)/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
