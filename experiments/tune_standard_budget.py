"""Standard-only development sweep; never trains defended/main study models.

Every budget uses the same predeclared targets and victim seeds. All outcomes,
including failures, are retained. Pilot selection is not a final ASR estimate.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison, load_state, write_json
from experiments.overnight_mnist import configuration, wilson


def run(args, root):
    keys = args.cases
    rows = []
    started = time.time()

    def config(output, budget, loss):
        value = configuration(output, keys, args.epochs, budget, loss)
        value.dataset, value.model, value.eps = args.dataset, args.model, args.eps
        value.precision = args.precision
        return value

    prep = Comparison(config(root/'preparation', args.budgets[0], args.losses[0]))
    if args.source:
        state = load_state(Path(args.source).expanduser())
        source = dict(reused_checkpoint=str(Path(args.source).expanduser().resolve()))
    else:
        source, state = prep.fit('surrogate', 'standard', 314159, 0.)
        threshold = .975 if args.dataset == 'MNIST' else .70
        if source['full_test_accuracy'] < threshold:
            raise RuntimeError('Source accuracy below development quality gate: '+str(source['full_test_accuracy']))
    identities = [(c['key'], c['target_id'], c['true_label'], c['intended_label']) for c in prep.cases]
    clean = {}
    for index, case in enumerate(prep.cases):
        record, _ = prep.fit('clean-'+str(index), 'standard', 60000+index, 0., case)
        clean[case['key']] = record
    write_json(root/'protocol.json', dict(dataset=args.dataset, model=args.model,
        parameters=sum(v.numel() for v in state.values()), epochs=args.epochs,
        training_precision=args.precision, poison_optimization_precision='fp32',
        source=source, cases=identities, victim_seeds=[60000+i for i in range(len(keys))],
        requested_budgets=args.budgets, losses=args.losses, epsilon=args.eps,
        attack_iterations=250, restarts=8, poison_batch=512,
        budget_definition='Fraction of all training images. Single-class budget is capped at the intended-label class size; actual counts are recorded.',
        selection='Smallest tested budget with >=2/3 intended-label successes and correct clean targets; development only. No main/defended runs.',
        git_commit=prep.manifest['git_commit']))

    def save(phase, **progress):
        write_json(root/'status.json', dict(phase=phase, elapsed_seconds=time.time()-started, **progress))
        write_json(root/'results.json', dict(development_only=True, dataset=args.dataset,
            model=args.model, epsilon=args.eps, rows=rows,
            warning='These cases are used to tune settings. Do not report the selected pilot ASR as an independent final estimate.'))

    for budget in args.budgets:
        for loss in args.losses:
            experiment = Comparison(config(root/('budget-'+str(budget)+'-'+loss), budget, loss))
            current = [(c['key'], c['target_id'], c['true_label'], c['intended_label']) for c in experiment.cases]
            if current != identities:
                raise RuntimeError('Targets changed across budgets; refuse unmatched comparison.')
            outcomes = []
            for index, case in enumerate(experiment.cases):
                save('brewing', budget=budget, loss=loss, case_index=index+1)
                delta = experiment.brew(case, state)
                save('training', budget=budget, loss=loss, case_index=index+1)
                record, _ = experiment.fit('standard-'+str(index), 'standard', 60000+index, 0., case, delta)
                target = next(t for t in record['targets'] if t['case_key']==case['key'])
                clean_target = next(t for t in clean[case['key']]['targets'] if t['case_key']==case['key'])
                outcomes.append(dict(case_key=case['key'], target_id=case['target_id'],
                    true_label=case['true_label'], intended_label=case['intended_label'],
                    actual_poison_count=len(case['poison_ids']),
                    actual_poison_fraction=len(case['poison_ids'])/len(experiment.labels),
                    target=target, clean_target=clean_target,
                    clean_test_accuracy=record['full_test_accuracy'],
                    clean_control_accuracy=clean[case['key']]['full_test_accuracy']))
                successes = sum(r['target']['attack_success'] for r in outcomes)
                row = dict(requested_budget=budget, loss=loss, successes=successes,
                    trials=len(outcomes), planned_trials=len(keys),
                    asr=successes/len(outcomes), asr_95ci=wilson(successes,len(outcomes)),
                    outcomes=list(outcomes))
                if rows and rows[-1]['requested_budget']==budget and rows[-1]['loss']==loss:
                    rows[-1]=row
                else:
                    rows.append(row)
                save('case_complete', budget=budget, loss=loss, case_index=index+1)
                print(json.dumps(dict(event='budget_case_complete', **row)), flush=True)
            del experiment
    viable = [r for r in rows if r['trials']>=3 and r['successes']/r['trials']>=2/3
              and all(o['clean_target']['target_correct'] for o in r['outcomes'])]
    selected = min(viable,key=lambda r:r['requested_budget']) if viable else None
    write_json(root/'selection.json', dict(viable=selected is not None, selected=selected,
        rule='Smallest tested budget with >=2/3 development success and correct clean targets; fresh held-out evaluation is still required.'))
    save('complete', viable=selected is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=['MNIST','CIFAR10'], default='MNIST')
    parser.add_argument('--model', default='LeNet')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eps', type=float, default=32.)
    parser.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    parser.add_argument('--budgets', type=float, nargs='+', default=[.06,.10])
    parser.add_argument('--losses', nargs='+', choices=['similarity','SE'], default=['similarity'])
    parser.add_argument('--cases', type=int, nargs='+', default=[1000000000,1100000000,1110000000])
    parser.add_argument('--source')
    args=parser.parse_args()
    if any(not 0 < b <= 1 for b in args.budgets) or args.epochs<1 or args.eps<=0:
        parser.error('Need positive budgets <=1, epochs, and epsilon.')
    if len(args.cases)!=len(set(args.cases)):
        parser.error('Development case keys must be distinct.')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    try:
        run(args,root)
    except BaseException:
        write_json(root/'status.json', dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
