"""Single-case standard-only activation intervention, with a freshly adapted attack.

Same layer dimensions, seed, data, schedule, target, poison IDs, budget and SE
attack settings as the saved Tanh failure. Only the model activation is ReLU.
This diagnostic is not a statistical ASR estimate or a defense comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, write_json
from experiments.overnight_mnist import configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/local-activation-test')
    parser.add_argument('--data',default='artifacts/local-data')
    args=parser.parse_args()
    root=Path(args.output).resolve()
    config=configuration(root,[3100000000],20,.04,'SE')
    config.model='LeNetReLU'
    config.data=str(Path(args.data).resolve())
    experiment=Comparison(config)
    expected=json.loads(Path('artifacts/local-inputs/mnist-original/main/manifest.json').read_text())
    expected=next(c for c in expected['cases'] if c['key']==3100000000)
    assert experiment.cases[0]==expected, 'Intervention changed target or poison subset'
    source,state=experiment.fit('surrogate','standard',314159,0.)
    if source['full_test_accuracy']<.975:
        raise RuntimeError('ReLU source did not reach the 97.5% clean accuracy gate')
    clean,_=experiment.fit('clean-victim','standard',70000,0.)
    started=time.perf_counter()
    delta=experiment.brew(experiment.cases[0],state)
    poison,_=experiment.fit('poison-victim','standard',70000,0.,experiment.cases[0],delta)
    write_json(root/'result.json',dict(intervention='Tanh -> ReLU only; poisons freshly optimized for this activation',
        parameters=sum(v.numel() for v in state.values()),dataset='MNIST',case_key=3100000000,
        epsilon=32/255,budget=.04,attack_iterations=250,restarts=8,loss='SE',
        source_accuracy=source['full_test_accuracy'],clean=clean,poison=poison,
        generation_and_victim_seconds=time.perf_counter()-started,
        warning='Single chosen diagnostic case; cannot estimate ASR or isolate every reason for the original 0/25.'))


if __name__=='__main__':
    main()
