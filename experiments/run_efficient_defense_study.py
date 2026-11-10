"""Run exactly one matched victim per case and method using a qualified pilot."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, METHODS, write_json
from experiments.search_efficient_conditions import configuration
from experiments.overnight_mnist import statistics


def run(args,root):
    selected=json.loads(Path(args.conditions).read_text())
    if not selected.get('qualified'):
        raise RuntimeError('This setting has not passed the standard-attack pilot.')
    trial_keys=[4000000000+1009*i for i in range(args.trials)]
    setup=argparse.Namespace(data=args.data,model=selected['model'],train_size=selected['train_size'],
        epochs=selected['epochs'],budget=selected['budget'],loss=selected['attack_loss'],
        restarts=selected['restarts'])
    config=configuration(setup,root/'main',trial_keys)
    if any(vars(config)[key]!=selected[key] for key in
           ('dataset','model','train_size','subset_seed','epochs','eps','budget','attack_loss',
            'attack_iterations','restarts','poison_batch','precision','radius','calibration_ratio')):
        raise RuntimeError('Study configuration differs from the qualified setting.')
    experiment=Comparison(config)
    started=time.time()
    source,state=experiment.fit('surrogate','standard',314159,0.)
    if source['full_test_accuracy']<.75:
        raise RuntimeError('Clean source below the 75% accuracy gate.')
    weights=experiment.calibrate(state)
    write_json(root/'protocol.json',dict(conditions=selected,trials=args.trials,methods=list(METHODS),
        total_victim_trainings=4*args.trials,case_to_seed={str(k):70000+i for i,k in enumerate(trial_keys)},
        source_accuracy=source['full_test_accuracy'],weights=weights,
        evaluation='Fixed poisons; not defense-aware. Fresh case keys, no success filtering.',
        comparison='Same poison, fresh victim seed, minibatch order and spatial draws within each case.',
        git_commit=experiment.manifest['git_commit']))
    for index,case in enumerate(experiment.cases):
        write_json(root/'status.json',dict(phase='brewing',case_index=index+1,trials=args.trials))
        delta=experiment.brew(case,state)
        for method in METHODS:
            write_json(root/'status.json',dict(phase='training',case_index=index+1,method=method,trials=args.trials))
            experiment.fit('trial-{:02d}-{}'.format(index,method),method,70000+index,weights[method],case,delta)
            statistics(root,args.trials)
    write_json(root/'status.json',dict(phase='complete',victim_trainings=4*args.trials,
                                    wall_seconds=time.time()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions',required=True)
    parser.add_argument('--output',default='artifacts/efficient-defense-25')
    parser.add_argument('--data',default='artifacts/local-data')
    parser.add_argument('--trials',type=int,default=25)
    args=parser.parse_args()
    if not 1<=args.trials<=1000:
        parser.error('Use 1 to 1000 trials.')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    try:
        run(args,root)
    except BaseException:
        write_json(root/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
