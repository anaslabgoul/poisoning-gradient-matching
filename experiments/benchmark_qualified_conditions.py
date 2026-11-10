"""Warm GPU kernels before estimating four-method study cost from short batches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, METHODS, load_state, write_json
from experiments.search_efficient_conditions import configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',required=True)
    args=parser.parse_args()
    root=Path(args.pilot).resolve()
    report=json.loads((root/'results.json').read_text())
    if not report.get('qualified') or not (root/'selected-conditions.json').exists():
        raise RuntimeError('Complete attack qualification before timing defenses.')
    setup=argparse.Namespace(**report['protocol']['config'])
    experiment=Comparison(configuration(setup,root/'validation',report['protocol']['validation_cases']))
    state=load_state(report['source']['checkpoint'])
    weights=experiment.calibrate(state)
    measured=[]
    for index,method in enumerate(METHODS):
        experiment.fit('warmup-'+method,method,90000+index,weights[method],max_batches=8)
        timing,_=experiment.fit('steady-timing-'+method,method,90000+index,weights[method],max_batches=60)
        measured.append(dict(method=method,batches=60,warmup_batches=8,
            measured_seconds=timing['train_seconds'],
            linear_projection_seconds=timing['train_seconds']/60*np.ceil(len(experiment.labels)/128)*setup.epochs,
            peak_gpu_gib=timing['peak_gpu_bytes']/1024**3))
    standard_measured=measured[0]['measured_seconds']
    validation=report['phases']['validation']['summary']
    for row in measured:
        row['projected_training_seconds']=validation['mean_victim_training_seconds']*row['measured_seconds']/standard_measured
    total=25*(validation['mean_poison_generation_seconds']+
              sum(t['projected_training_seconds'] for t in measured))+report['source']['training_seconds']
    timing_report=dict(methods=measured,projected_25case_4method_seconds=total,
        method='Eight warmup batches, 60 timed batches per method; ratios anchored to measured full standard-model training.',
        warning='Time estimate, not a completed defense comparison; excludes future interruptions and some checkpoint/summary overhead.',
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'steady-timing.json',timing_report)
    report['initial_defense_timing']=report.get('defense_timing')
    report['defense_timing']=measured
    report['projected_25case_4method_seconds']=total
    report['timing_method']=timing_report['method']
    write_json(root/'results.json',report)
    print(json.dumps(timing_report,indent=2),flush=True)


if __name__=='__main__':
    main()
