"""Qualify a cheap standard-training attack on development and fresh pilot cases.

Only standard models are fully trained. Defenses are timed on short batches
after qualification; a 25-case study remains a separate, independent experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, METHODS, load_state, write_json
from experiments.overnight_mnist import wilson
from forest.victims.models import get_model


def configuration(args, output, cases):
    return argparse.Namespace(output=str(Path(output).resolve()),data=str(Path(args.data).resolve()),
        dataset='CIFAR10',model=args.model,train_size=args.train_size,subset_seed=20261110,
        cases=cases,seeds=[60000],clean_controls=False,surrogate_seed=314159,
        epochs=args.epochs,attack_iterations=250,restarts=args.restarts,poison_batch=512,
        eps=32.,budget=args.budget,radius=32.,calibration_ratio=.1,
        precision='bf16',threads=4,benchmark=0,attack_loss=args.loss)


def summarize(rows):
    if not rows:
        return None
    n=len(rows)
    success=sum(r['poison_target']['attack_success'] for r in rows)
    natural=sum(r['clean_target']['attack_success'] for r in rows)
    attributable=sum(r['poison_target']['attack_success'] and r['clean_target']['target_correct'] for r in rows)
    return dict(trials=n,successes=success,asr=success/n,asr_95ci=wilson(success,n),
        clean_control_intended_successes=natural,attributable_successes=attributable,
        mean_clean_control_accuracy=float(np.mean([r['clean_accuracy'] for r in rows])),
        mean_poisoned_model_accuracy=float(np.mean([r['poison_accuracy'] for r in rows])),
        mean_victim_training_seconds=float(np.mean([r['victim_training_seconds'] for r in rows])),
        mean_poison_generation_seconds=float(np.mean([r['poison_generation_seconds'] for r in rows])),
        peak_gpu_gib=max(max(r['peak_gpu_bytes'],r['poison_generation_peak_gpu_bytes']) for r in rows)/1024**3)


def run(args,root):
    dev_cases=[2000000000,2100000000,2110000000]
    heldout_cases=[3200000000+args.validation_offset+1009*i for i in range(5)]
    started=time.time()
    protocol=dict(config=vars(args),development_cases=dev_cases,validation_cases=heldout_cases,
        subset='Fixed balanced CIFAR-10 training subset; original 10,000-image test set retained.',
        primary='Exact intended wrong label, over all predeclared attack cases.',
        attribution='Also require successes on targets that the matched clean control classified correctly.',
        development_gate='At least 2 attributable successes in 3 cases; reject early only when impossible.',
        validation_gate='5 cases: >=4 intended-label successes, >=3 attributable successes, and >=75% mean accuracy.',
        clean_quality_gate=.75,
        statement='This is a configuration-search pilot, not a final unbiased ASR estimate. No targets are filtered by attack outcomes.')
    write_json(root/'protocol.json',protocol)
    report=dict(protocol=protocol,phases={},qualified=False)

    def save(phase,**progress):
        write_json(root/'status.json',dict(phase=phase,elapsed_seconds=time.time()-started,**progress))
        write_json(root/'results.json',report)

    dev=Comparison(configuration(args,root/'development',dev_cases))
    save('source_training')
    if args.source:
        state=load_state(args.source)
        source_record=json.loads(Path(args.source_record).read_text())
        previous=json.loads((Path(args.source).parents[1]/'manifest.json').read_text())
        for key in ('dataset','model','train_size','subset_seed','epochs','precision','surrogate_seed'):
            if previous['config'].get(key)!=vars(dev.config).get(key):
                raise RuntimeError('Source reuse mismatch: '+key)
        if previous['training_subset']['original_indices']!=dev.manifest['training_subset']['original_indices']:
            raise RuntimeError('Source subset mismatch')
    else:
        source_record,state=dev.fit('surrogate','standard',314159,0.)
    report['source']=dict(accuracy=source_record['full_test_accuracy'],
        training_seconds=source_record['train_seconds'],
        parameters=sum(p.numel() for p in get_model(args.model,'CIFAR10').parameters()),
        checkpoint=str((Path(args.source) if args.source else dev.out/'models/surrogate.pt').resolve()))
    if source_record['full_test_accuracy']<.75:
        report['rejection']='Source accuracy below 75%; no attack pilot launched.'
        save('rejected_source')
        return

    for phase,keys,seed_base in [('development',dev_cases,60000),('validation',heldout_cases,61000)]:
        experiment=dev if phase=='development' else Comparison(configuration(args,root/'validation',keys))
        rows=[]
        report['phases'][phase]=dict(rows=rows,summary=None)
        for index,case in enumerate(experiment.cases):
            save('clean_control',stage=phase,case_index=index+1)
            clean,_=experiment.fit('clean-'+str(index),'standard',seed_base+index,0.)
            save('poison_generation',stage=phase,case_index=index+1)
            delta=experiment.brew(case,state)
            poison_meta=json.loads((experiment.out/'poisons'/(str(case['key'])+'.json')).read_text())
            save('poisoned_victim',stage=phase,case_index=index+1)
            poison,_=experiment.fit('poison-'+str(index),'standard',seed_base+index,0.,case,delta)
            target=lambda record:next(t for t in record['targets'] if t['case_key']==case['key'])
            rows.append(dict(case_key=case['key'],seed=seed_base+index,poison_count=len(delta),
                clean_target=target(clean),poison_target=target(poison),
                clean_accuracy=clean['full_test_accuracy'],poison_accuracy=poison['full_test_accuracy'],
                victim_training_seconds=poison['train_seconds'],
                poison_generation_seconds=poison_meta['seconds'],
                poison_generation_peak_gpu_bytes=poison_meta['peak_gpu_bytes'],
                peak_gpu_bytes=poison['peak_gpu_bytes'],matching_loss=poison_meta['matching_loss']))
            report['phases'][phase]['summary']=summarize(rows)
            save('case_complete',stage=phase,case_index=index+1)
            print(json.dumps(dict(event='pilot_case_complete',stage=phase,**rows[-1])),flush=True)
            attributable=report['phases'][phase]['summary']['attributable_successes']
            if phase=='development' and attributable+(3-len(rows))<2:
                report['rejection']='Development cannot reach 2/3 attributable success; later targets not tested.'
                save('rejected_development')
                return
        if phase=='development':
            if report['phases'][phase]['summary']['attributable_successes']<2:
                report['rejection']='Development gate failed.'
                save('rejected_development')
                return
        else:
            summary=report['phases'][phase]['summary']
            report['qualified']=(summary['successes']>=4 and summary['attributable_successes']>=3
                and summary['mean_poisoned_model_accuracy']>=.75)
            if not report['qualified']:
                report['rejection']='Fresh five-case validation did not meet the declared gate.'
                save('rejected_validation')
                return
            save('timing_defenses')
            weights=experiment.calibrate(state)
            timings=[]
            for index,method in enumerate(METHODS):
                timing,_=experiment.fit('timing-'+method,method,90000+index,weights[method],max_batches=30)
                projected=timing['train_seconds']/30*int(np.ceil(len(experiment.labels)/128))*args.epochs
                timings.append(dict(method=method,batches=30,measured_seconds=timing['train_seconds'],
                    projected_training_seconds=projected,peak_gpu_gib=timing['peak_gpu_bytes']/1024**3))
            report['defense_timing']=timings
            report['projected_25case_4method_seconds']=(25*(summary['mean_poison_generation_seconds']+
                sum(t['projected_training_seconds'] for t in timings))+source_record['train_seconds'])
            selected=dict(qualified=True,model=args.model,train_size=args.train_size,subset_seed=20261110,
                dataset='CIFAR10',epochs=args.epochs,eps=32.,budget=args.budget,
                attack_loss=args.loss,attack_iterations=250,restarts=args.restarts,poison_batch=512,
                precision='bf16',radius=32.,calibration_ratio=.1,source_seed=314159,
                validation_summary=summary,pilot_report=str((root/'results.json').resolve()),
                instruction='Use fresh case keys and one matched victim seed per case for the independent defense study.')
            write_json(root/'selected-conditions.json',selected)
    save('complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data',default='artifacts/local-data')
    parser.add_argument('--model',default='ResNet20')
    parser.add_argument('--train-size',type=int,default=10000)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--budget',type=float,default=.04)
    parser.add_argument('--loss',choices=['similarity','SE'],default='similarity')
    parser.add_argument('--restarts',type=int,default=4)
    parser.add_argument('--validation-offset',type=int,default=0)
    parser.add_argument('--source')
    parser.add_argument('--source-record')
    args=parser.parse_args()
    if bool(args.source)!=bool(args.source_record):
        parser.error('Source reuse requires both --source and --source-record')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    try:
        run(args,root)
    except BaseException:
        write_json(root/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
