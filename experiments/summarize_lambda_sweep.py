"""Collect saved matched lambda runs without choosing a winner on test outcomes."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import write_json
from experiments.compare_saved_gradient import aggregate
from experiments.fixed_target_grid import sha256


def collect(plan_path,output,cohort_path):
    plan=json.loads(Path(plan_path).read_text())
    cohort=json.loads(Path(cohort_path).read_text())
    root=Path(output)
    if cohort['expected']!=plan['runs_per_lambda']:
        raise ValueError('Sweep cohort count mismatch')
    rows=[]
    for setting in plan['settings']:
        directory=root/setting['directory']
        records=[json.loads(p.read_text()) for p in (directory/'completed').glob('*.json')]
        if any(not math.isclose(r['lambda_gradient'],setting['lambda'],rel_tol=1e-12,abs_tol=0.) for r in records):
            raise ValueError('Outcome belongs to a different lambda')
        result=aggregate(cohort,records)
        result.pop('records')
        status_path=directory/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {'phase':'pending'}
        rows.append(dict(**setting,**result,status=status))
    reference=plan['existing_reference']
    path=Path(reference['comparison'])
    if sha256(path)!=reference['comparison_sha256']:
        raise ValueError('Original lambda result changed')
    original=json.loads(path.read_text())
    if any(not math.isclose(r['lambda_gradient'],reference['lambda'],rel_tol=1e-12,abs_tol=0.) for r in original['records']):
        raise ValueError('Reference coefficient mismatch')
    # Match original outcomes to this same frozen baseline, independently.
    checked=aggregate(cohort,original['records'])
    checked.pop('records')
    reused=dict(lambda_value=reference['lambda'],source=str(path),**checked)
    completed=sum(r['completed'] for r in rows)
    result=dict(complete=all(r['complete'] for r in rows),
        completed_new_runs=completed,expected_new_runs=plan['expected_new_runs'],
        settings=rows,existing_reference=reused,
        baseline=dict(successes=cohort['baseline_successes'],trials=cohort['expected'],
                      asr=cohort['baseline_asr'],clean_accuracy=cohort['baseline_accuracy']),
        interpretation=plan['interpretation'])
    write_json(root/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='experiments/protocols/gradient-lambda-sweep.json')
    parser.add_argument('--cohort',default='experiments/protocols/gradient-defense-cohort-25.json')
    parser.add_argument('--output',default='artifacts/gradient-lambda-sweep')
    args=parser.parse_args()
    result=collect(args.plan,args.output,args.cohort)
    print(json.dumps(dict(complete=result['complete'],completed_new_runs=result['completed_new_runs'],
                         expected_new_runs=result['expected_new_runs'])),flush=True)


if __name__=='__main__':
    main()
