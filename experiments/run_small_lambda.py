"""Explicitly requested fixed-lambda batches using the saved poison and baseline."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import load_state,write_json
from experiments.run_bilevel_trials import BilevelTrials,SliceExpired,make_config
from experiments.run_bilevel_overnight import slice_outcome


def verify_inputs(protocol,source):
    for relative,expected in protocol['frozen_files'].items():
        path=source/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source file changed: '+relative)
    manifest=json.loads((source/'manifest.json').read_text())
    baseline=json.loads((source/'runs/trial-0-standard.json').read_text())
    expected=dict(dataset='CIFAR10',model='ResNet18',train_size=None,epochs=40,batch_size=128,
        cases=protocol['case_keys'],seeds=protocol['victim_seeds'],precision='fp32',
        eps=16.,budget=.01,radius=16.,inner_steps=3,regularizer_samples=8)
    if any(manifest['config'].get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Saved baseline settings differ from the requested matched experiment.')
    if (baseline['status']!='complete' or baseline['seed']!=protocol['victim_seeds'][0] or
        baseline['case_key']!=protocol['case_keys'][0] or baseline['poison_sha256']!=
        protocol['frozen_files']['poisons/3800000000.pt']):
        raise RuntimeError('Saved baseline outcome identity mismatch.')
    return manifest,baseline


def validate_protocol(protocol):
    allowed=([1e-8,1e-10],[1e-7,1e-6,1e-5])
    count=len(protocol['lambdas'])
    if (protocol['lambdas'] not in allowed or protocol['trial_count']!=1 or
        protocol['new_victim_trainings']!=count or protocol['victim_models']!=count+1 or
        protocol['reused_undefended_models']!=1 or protocol['additional_clean_surrogates']!=0 or
        protocol['dataset']!='CIFAR10' or protocol['model']!='ResNet18' or
        protocol['train_images']!=50000 or protocol['test_images']!=10000 or
        protocol['epochs']!=40 or protocol['batch_size']!=128 or
        protocol['case_keys']!=[3800000000] or protocol['victim_seeds']!=[18001] or
        protocol['defense']['radius_pixel_units']!=16. or protocol['defense']['inner_steps']!=3 or
        protocol['defense']['regularizer_samples_per_batch']!=8 or
        protocol['optimizer']!=dict(learning_rate=.1,momentum=.9,nesterov=True,
            weight_decay=.0005,clipping=False,warmup=False,adaptive_lambda=False,update_rescaling=False)):
        raise ValueError('Require an explicitly requested lambda batch and the unchanged one-case benchmark.')


class SmallLambdaTrials(BilevelTrials):
    def __init__(self,config,protocol,slice_seconds,source,baseline):
        self.source=source
        self.baseline=baseline
        super().__init__(config,protocol,slice_seconds)

    def poison_path(self,case):
        return self.source/'poisons'/(str(case['key'])+'.pt')

    def run_cached(self):
        base=copy.deepcopy(self.baseline)
        base['reused_from']=str(self.source/'runs/trial-0-standard.json')
        base['cached_model_sha256']=self.protocol['frozen_files']['models/trial-0-standard.pt']
        write_json(self.out/'runs/trial-0-standard.json',base)
        self.summarize_trials()
        case=self.cases[0]
        poison=load_state(self.poison_path(case))
        if poison['poison_ids'].tolist()!=case['poison_ids'] or int(poison['target_id'])!=case['target_id']:
            raise RuntimeError('Poison target or training row alignment changed.')
        write_json(self.out/'inputs_verified.json',dict(frozen_files=self.protocol['frozen_files'],
            baseline_reused=True,new_poison_generation=False,source_retraining=False,
            optimizer=self.protocol['optimizer'],regularizer_samples=8))
        for weight in self.protocol['lambdas']:
            self.train_condition('trial-0-lambda-'+str(weight),self.config.seeds[0],weight,case,poison['delta'])
            self.summarize_trials()
        final=self.summarize_trials()
        write_json(self.out/'status.json',dict(phase='complete' if final['all_succeeded'] else 'complete_with_failures',
            attempted=final['attempted'],expected=self.protocol['victim_models'],
            new_defended_trainings=self.protocol['new_victim_trainings'],cached_baselines=1))


def main(default_protocol='experiments/protocols/bilevel-small-lambda.json',
         default_output='artifacts/bilevel-small-lambda'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default=default_protocol)
    parser.add_argument('--output',default=default_output)
    parser.add_argument('--data',default='~/data')
    parser.add_argument('--slice-seconds',type=float,default=6000.)
    args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text())
    validate_protocol(protocol)
    root=Path(args.output)
    hard_stop=datetime.fromisoformat(protocol['hard_stop_utc']).timestamp()
    if time.time()>=hard_stop-180:
        write_json(root/'status.json',dict(phase='deadline_reached',hard_stop_utc=protocol['hard_stop_utc']))
        return 0
    try:
        source=Path(protocol['baseline_directory']).expanduser().resolve()
        manifest,baseline=verify_inputs(protocol,source)
        if torch.__version__!=manifest['torch']:
            raise RuntimeError('PyTorch version differs from the cached baseline: '+torch.__version__)
        config=make_config(args,protocol)
        config.fixed_cases=manifest['cases']
        config.regularizer_samples=8
        config.gradient_diagnostics=True
        config.validate_buffers=True
        config.baseline_source=str(source)
        experiment=SmallLambdaTrials(config,protocol,args.slice_seconds,source,baseline)
        experiment.deadline=min(experiment.deadline,time.monotonic()+max(0,hard_stop-time.time()))
        experiment.run_cached()
    except SliceExpired:
        phase,code=slice_outcome(hard_stop,time.time())
        write_json(root/'status.json',dict(phase=phase,hard_stop_utc=protocol['hard_stop_utc'],
            reason='Checkpointed; continue only before the morning cutoff.'))
        return code
    except BaseException:
        write_json(root/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise
    return 0


if __name__=='__main__':
    sys.exit(main())
