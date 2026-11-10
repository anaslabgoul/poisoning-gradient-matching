"""Short sanity controls for the failed MNIST baseline; no defense study."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import Comparison,load_state,write_json
from experiments.overnight_mnist import configuration
from forest.victims.models import get_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='artifacts/mnist-lenet-25-study')
    parser.add_argument('--output',default='artifacts/mnist-baseline-diagnostics')
    args=parser.parse_args()
    study=Path(args.study)
    manifest=json.loads((study/'main/manifest.json').read_text())
    report=json.loads((study/'statistics.json').read_text())
    dataset=torchvision.datasets.MNIST(str(Path('~/data').expanduser()),train=True,download=False)
    audit=[]
    for case in manifest['cases']:
        saved=load_state(study/'main/poisons'/(str(case['key'])+'.pt'))
        ids=saved['poison_ids']
        delta=saved['delta']
        assert ids.tolist()==case['poison_ids']
        assert int(saved['target_id'])==case['target_id']
        assert bool((dataset.targets[ids]==case['intended_label']).all())
        assert case['true_label']!=case['intended_label']
        assert torch.isfinite(delta).all()
        linf=float(delta.abs().max())*.30810782313346863
        assert linf<=32/255+1e-6
        audit.append(dict(case=case['key'],count=len(ids),pixel_linf=linf,
                          nonzero=int(torch.count_nonzero(delta))))
    key=manifest['cases'][0]['key']
    experiment=Comparison(configuration(args.output,[key],20,.04,'SE'))
    case=experiment.cases[0]
    saved=load_state(study/'main/poisons'/(str(key)+'.pt'))
    assert saved['poison_ids'].tolist()==case['poison_ids']
    delta=saved['delta']
    ids=saved['poison_ids'].to(experiment.device)
    original=experiment.images[ids]
    poisoned=original+delta.to(experiment.device)
    changed=int((poisoned-original).flatten(1).abs().sum(1).gt(0).sum())
    assert changed==len(ids)
    assert float((poisoned*experiment.ds+experiment.dm).min())>=-1e-6
    assert float((poisoned*experiment.ds+experiment.dm).max())<=1+1e-6

    model=get_model('LeNet','MNIST').to(experiment.device)
    model.load_state_dict(load_state(study/'preparation/models/surrogate.pt'))
    model.eval()
    target=experiment.test_images[case['target_id']:case['target_id']+1]
    adversarial_label=torch.tensor([case['intended_label']],device=experiment.device)
    target_grad=torch.autograd.grad(F.cross_entropy(model(target),adversarial_label),model.parameters())
    target_vector=torch.cat([g.flatten() for g in target_grad]).detach()
    gradient_diagnostics={}
    labels=experiment.labels[ids]
    for name,images in [('clean',original),('poisoned',poisoned)]:
        outputs=model(images)
        gradient=torch.autograd.grad(F.cross_entropy(outputs,labels),model.parameters())
        vector=torch.cat([g.flatten() for g in gradient]).detach()
        gradient_diagnostics[name]=dict(target_gradient_norm=float(target_vector.norm()),
            poison_gradient_norm=float(vector.norm()),
            cosine=float(F.cosine_similarity(vector,target_vector,dim=0)),
            squared_distance=float((vector-target_vector).square().sum()),
            source_poison_image_accuracy=float((outputs.argmax(1)==labels).float().mean()))
    del model
    torch.cuda.empty_cache()

    clean,clean_state=experiment.fit('clean-new-init','standard',70000,0.)
    same_init,_=experiment.fit('valid-poison-source-init','standard',314159,0.,case,delta)
    # Deliberately outside the attack constraints: replace every selected base
    # image by the exact target, retaining the attacker's label. This positive
    # control verifies the loader/training/target-scoring path, not attack efficacy.
    dirty_delta=(target-original).cpu()
    positive,_=experiment.fit('UNBOUNDED-positive-control','standard',70000,0.,case,dirty_delta)
    baseline=next(r for r in report['records'] if r['case_key']==key and r['method']=='standard')
    baseline_state=load_state(study/'main/models'/(baseline['name']+'.pt'))
    distance=sum(float((baseline_state[k]-v).square().sum()) for k,v in clean_state.items())**.5
    def outcome(record):
        target_result=next(t for t in record['targets'] if t['case_key']==key)
        return dict(accuracy=record['full_test_accuracy'],target=target_result)
    result=dict(audited_cases=len(audit),audit=audit,checked_case=case,
        changed_training_rows=changed,gradient_diagnostics=gradient_diagnostics,
        controls=dict(clean_new_initialization=outcome(clean),
                      bounded_poison_new_initialization=outcome(baseline),
                      bounded_poison_source_initialization=outcome(same_init),
                      unbounded_target_insertion_positive_control=outcome(positive)),
        poisoned_vs_clean_parameter_l2=distance,
        warning='Positive control violates the epsilon bound and clean-label semantic constraint; it is only a pipeline diagnostic.')
    write_json(Path(args.output)/'diagnosis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
