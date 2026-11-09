"""Persistent MNIST/LeNet study: 25 independent matched cases x four methods.

Development cases are disjoint from reported cases. Attack budget is selected
on development only, using a predeclared 1% then 4% grid. All reported cases
are retained, including unsuccessful attacks. No defense-aware attack is used.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, METHODS, write_json


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def mean_ci(values, seed=12345):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    draws = np.random.default_rng(seed).integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def statistics(root, trials):
    directory = root/'main'
    records = [json.loads(p.read_text()) for p in (directory/'runs').glob('trial-*.json')]
    summaries = []
    baseline = {r['case_key']: r for r in records if r['method']=='standard'}
    for method in METHODS:
        rows = sorted([r for r in records if r['method']==method], key=lambda r:r['case_key'])
        if not rows:
            continue
        outcomes = [next(t for t in r['targets'] if t['case_key']==r['case_key']) for r in rows]
        accuracy = [r['full_test_accuracy'] for r in rows]
        group = [t['unseen_group_intended_rate'] for t in outcomes]
        successes = sum(r['attack_success'] for r in rows)
        paired = [r for r in rows if r['case_key'] in baseline]
        attack_delta = [r['attack_success']-baseline[r['case_key']]['attack_success'] for r in paired]
        accuracy_delta = [r['full_test_accuracy']-baseline[r['case_key']]['full_test_accuracy'] for r in paired]
        summaries.append(dict(method=method, trials=len(rows), successes=successes,
            targeted_asr=successes/len(rows), targeted_asr_95ci=wilson(successes,len(rows)),
            mean_clean_test_accuracy=float(np.mean(accuracy)),
            accuracy_sd=float(np.std(accuracy,ddof=1)) if len(rows)>1 else None,
            accuracy_mean_95ci=mean_ci(accuracy),
            mean_unseen_source_class_wrong_label_rate=float(np.mean(group)),
            source_class_rate_mean_95ci=mean_ci(group),
            paired_asr_difference_vs_standard=float(np.mean(attack_delta)),
            paired_asr_difference_95ci=mean_ci(attack_delta),
            paired_accuracy_difference_vs_standard=float(np.mean(accuracy_delta)),
            paired_accuracy_difference_95ci=mean_ci(accuracy_delta),
            mean_training_seconds=float(np.mean([r['train_seconds'] for r in rows])),
            peak_gpu_gib=max(r['peak_gpu_bytes'] for r in rows)/1024**3))
    complete=len(summaries)==4 and all(r['trials']==trials for r in summaries)
    result=dict(complete=complete, expected_trials_per_method=trials,
        targeted_metric='Fraction of independent single-target poisoning cases attaining the exact intended wrong label.',
        group_metric='Within each case, fraction of OTHER test images of the source class predicted as intended label; then average across cases.',
        uncertainty='Wilson 95% intervals for targeted ASR; 10,000 case-resampling bootstrap draws for means and paired differences. Descriptive; no multiple-comparison significance claim.',
        summaries=summaries, completed_victim_trainings=len(records), records=records)
    write_json(root/'statistics.json', result)
    return result


def configuration(output, cases, epochs, budget):
    return argparse.Namespace(output=str(output),data='~/data',dataset='MNIST',model='LeNet',
        cases=cases,seeds=[70000],clean_controls=False,surrogate_seed=314159,
        epochs=epochs,attack_iterations=250,restarts=8,poison_batch=512,eps=32.,
        budget=budget,radius=32.,calibration_ratio=.1,precision='fp32',threads=4,benchmark=0)


def study(root, trials, epochs):
    dev_cases=[1000000000,1100000000,1110000000]
    main_cases=[3100000000+1009*i for i in range(trials)]
    started=time.time()

    def status(phase, **values):
        write_json(root/'status.json',dict(phase=phase,updated_unix=time.time(),**values))

    status('preparing',expected_victim_trainings=4*trials)
    prep=Comparison(configuration(root/'preparation',dev_cases,epochs,.01))
    source_result,state=prep.fit('surrogate','standard',314159,0.)
    accuracy=source_result['full_test_accuracy']
    if accuracy < .975:
        raise RuntimeError('Clean LeNet accuracy below 97.5% quality gate: '+str(accuracy))
    weights=prep.calibrate(state)
    parameter_count=sum(v.numel() for k,v in state.items() if not k.endswith('num_batches_tracked'))
    selection_path=root/'attack-selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
    else:
        development=[]
        for budget in [.01,.04]:
            status('development',poison_budget=budget,source_accuracy=accuracy)
            dev=Comparison(configuration(root/('development-'+str(budget)),dev_cases,epochs,budget))
            outcomes=[]
            for i,case in enumerate(dev.cases):
                delta=dev.brew(case,state)
                result,_=dev.fit('development-'+str(i),'standard',60000+i,0.,case,delta)
                outcomes.append(result['attack_success'])
            development.append(dict(budget=budget,successes=sum(outcomes),trials=3,outcomes=outcomes))
            del dev
            if sum(outcomes)>=1:
                break
        selection=dict(selected_budget=development[-1]['budget'],development=development,
            rule='First budget in [0.01,0.04] with at least one success in three disjoint development cases; otherwise retain 0.04 and flag weak baseline.',
            weak_development_baseline=sum(development[-1]['outcomes'])==0)
        write_json(selection_path,selection)
    main=Comparison(configuration(root/'main',main_cases,epochs,selection['selected_budget']))
    for filename in ['weights.json','calibration.json']:
        if not (main.out/filename).exists():
            shutil.copyfile(prep.out/filename,main.out/filename)
    write_json(root/'protocol.json',dict(dataset='MNIST',image_shape=[1,28,28],model='LeNet',
        parameters=parameter_count,epochs=epochs,trials_per_method=trials,methods=list(METHODS),
        case_to_victim_seed={str(k):70000+i for i,k in enumerate(main_cases)},
        clean_source_seed=314159,clean_source_accuracy=accuracy,
        matched='Same poison tensor, initialization, minibatch order and spatial augmentation draws per case.',
        augmentation='Differentiable translation up to approximately two pixels; no flips for digits.',
        attack_selection=selection,weights=weights,precision='FP32; TF32 CUDA math enabled',
        primary_metric='Single-target success averaged over 25 independent cases (paper style).',
        secondary_metric='Source-class wrong-label transfer on all other images, independently per case.',
        attacks='Fixed gradient-matching poisons, not defense-aware. Epsilon 32/255, 250 iterations, eight restarts.',
        git_commit=main.manifest['git_commit']))
    initial_summary=statistics(root,trials)
    for index,case in enumerate(main.cases):
        status('brewing',case_index=index+1,trials=trials,
               completed_victim_trainings=initial_summary['completed_victim_trainings'])
        delta=main.brew(case,state)
        for method in METHODS:
            status('training',case_index=index+1,trials=trials,method=method)
            main.fit('trial-{:02d}-{}'.format(index,method),method,70000+index,weights[method],case,delta)
            initial_summary=statistics(root,trials)
        print(json.dumps(dict(event='case_complete',case_index=index+1,
                              summary=initial_summary['summaries'])),flush=True)
    status('complete',completed_victim_trainings=4*trials,total_wall_seconds=time.time()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/mnist-lenet-25')
    parser.add_argument('--trials',type=int,default=25)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if args.trials<1 or args.epochs<1:
        parser.error('Need positive trial and epoch counts.')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    try:
        study(root,args.trials,args.epochs)
    except BaseException:
        write_json(root/'status.json',dict(phase='failed',traceback=traceback.format_exc(),updated_unix=time.time()))
        raise


if __name__=='__main__':
    main()
