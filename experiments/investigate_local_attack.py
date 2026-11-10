"""Inspect saved poisons locally and trace matched standard-training controls.

No defense models or new attack cases are generated. Saved perturbations,
normalization, target IDs and labels are audited before measuring gradients.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_defenses import load_state, seed_all, write_json
from forest.consts import mnist_mean, mnist_std, cifar10_mean, cifar10_std
from forest.data.diff_data_augmentation import RandomTransform
from forest.victims.models import get_model


def model_for(name, dataset, state, device):
    model=get_model(name,dataset).to(device,memory_format=torch.channels_last)
    model.load_state_dict(state)
    return model.eval()


def data_for(dataset, root, device):
    constructor=getattr(torchvision.datasets,dataset)
    train=constructor(root=str(root),train=True,download=True)
    test=constructor(root=str(root),train=False,download=True)
    mean,std=(mnist_mean,mnist_std) if dataset=='MNIST' else (cifar10_mean,cifar10_std)
    dm=torch.tensor(mean,device=device)[None,:,None,None]
    ds=torch.tensor(std,device=device)[None,:,None,None]
    def convert(data):
        raw=torch.as_tensor(data,device=device).float()/255
        raw=raw[:,None] if raw.ndim==3 else raw.permute(0,3,1,2)
        return ((raw-dm)/ds).contiguous(memory_format=torch.channels_last)
    return dict(images=convert(train.data),labels=torch.as_tensor(train.targets,device=device),
                test_images=convert(test.data),test_labels=torch.as_tensor(test.targets,device=device),
                dm=dm,ds=ds)


def target_result(model, target, intended, true_label):
    with torch.no_grad():
        logits=model(target).float()
        probabilities=logits.softmax(1)[0]
        return dict(predicted_label=int(logits.argmax(1)),true_label=int(true_label),
                    intended_label=int(intended),attack_success=int(logits.argmax(1)==intended),
                    intended_probability=float(probabilities[intended]),
                    intended_loss=float(F.cross_entropy(logits,torch.tensor([intended],device=target.device))),
                    intended_minus_true_logit=float(logits[0,intended]-logits[0,true_label]))


def gradient_vector(model, images, labels, batch=128):
    # All inspected models are evaluated with fixed normalization statistics.
    # Chunking is a weighted exact mean, not a change of poison batch objective.
    vector=torch.zeros(sum(p.numel() for p in model.parameters()),device=images.device)
    correct=0
    for start in range(0,len(images),batch):
        x,y=images[start:start+batch],labels[start:start+batch]
        outputs=model(x)
        gradients=torch.autograd.grad(F.cross_entropy(outputs,y),model.parameters())
        vector.add_(torch.cat([g.flatten() for g in gradients]),alpha=len(x)/len(images))
        correct+=int((outputs.argmax(1)==y).sum())
    return vector,correct/len(images)


def audit_and_align(case, saved, states, model_name, dataset, data):
    device=data['images'].device
    ids=saved['poison_ids'].to(device)
    delta=saved['delta'].to(device)
    assert ids.tolist()==case['poison_ids'], 'Poison ID/order mismatch'
    assert int(saved['target_id'])==case['target_id'], 'Target ID mismatch'
    original=data['images'][ids]
    poisoned=original+delta
    pixel_delta=delta*data['ds']
    pixels=poisoned*data['ds']+data['dm']
    assert torch.isfinite(delta).all()
    assert float(pixel_delta.abs().max())<=32/255+1e-6
    assert float(pixels.min())>=-1e-6 and float(pixels.max())<=1+1e-6
    labels=data['labels'][ids]
    assert bool((labels==case['intended_label']).all())
    target=data['test_images'][case['target_id']:case['target_id']+1]
    assert int(data['test_labels'][case['target_id']])==case['true_label']
    result=dict(case_key=case['key'],dataset=dataset,model=model_name,
                target_id=case['target_id'],true_label=case['true_label'],intended_label=case['intended_label'],
                count=len(ids),fraction=len(ids)/len(data['labels']),
                changed_rows=int(delta.flatten(1).abs().sum(1).gt(0).sum()),
                pixel_linf=float(pixel_delta.abs().max()),states={})
    for name,state in states.items():
        model=model_for(model_name,dataset,state,device)
        intended=torch.tensor([case['intended_label']],device=device)
        target_gradient,_=gradient_vector(model,target,intended)
        clean_gradient,clean_accuracy=gradient_vector(model,original,labels)
        poison_gradient,poison_accuracy=gradient_vector(model,poisoned,labels)
        measures={}
        for kind,gradient in [('clean',clean_gradient),('poisoned',poison_gradient)]:
            measures[kind]=dict(norm=float(gradient.norm()),
                cosine=float(F.cosine_similarity(gradient,target_gradient,dim=0)),
                dot_target=float(gradient@target_gradient),
                squared_distance=float((gradient-target_gradient).square().sum()))
        before=target_result(model,target,case['intended_label'],case['true_label'])
        # A small poison-only gradient step tests the claimed *local* mechanism.
        # It is not a full-data training run or a valid attack success estimate.
        step_size=1e-4
        offset=0
        with torch.no_grad():
            for parameter in model.parameters():
                count=parameter.numel()
                parameter.add_(poison_gradient[offset:offset+count].view_as(parameter),alpha=-step_size)
                offset+=count
        after=target_result(model,target,case['intended_label'],case['true_label'])
        result['states'][name]=dict(target=before,target_gradient_norm=float(target_gradient.norm()),
            alignment=measures,clean_poison_subset_accuracy=clean_accuracy,
            perturbed_poison_subset_accuracy=poison_accuracy,
            poison_increment_dot_target=float((poison_gradient-clean_gradient)@target_gradient),
            one_poison_only_step=dict(step_size=step_size,target=after,
                intended_loss_change=after['intended_loss']-before['intended_loss']))
        del model
    return result


def inspect(args,root,device):
    source=Path(args.inputs)
    data=data_for('MNIST',Path(args.data),device)
    manifest=json.loads((source/'mnist-original/main/manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['key']==3100000000)
    seed_all(70000)
    states=dict(source_trained=load_state(source/'mnist-original/preparation/models/surrogate.pt'),
        victim_initial=copy.deepcopy(get_model('LeNet','MNIST').state_dict()),
        victim_clean_trained=load_state(source/'mnist-baseline-diagnostics/models/clean-new-init.pt'),
        victim_poison_trained=load_state(source/'mnist-original/main/models/trial-00-standard.pt'))
    result=dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                torch=torch.__version__,cases=[])
    result['cases'].append(audit_and_align(case,load_state(source/'mnist-original/main/poisons/3100000000.pt'),
                                            states,'LeNet','MNIST',data))
    write_json(root/'alignment.json',result)
    del data,states
    torch.cuda.empty_cache()
    cifar=source/'cifar-convnet32-standard-budget-01-bf16'
    data=data_for('CIFAR10',Path(args.data),device)
    manifest=json.loads((cifar/'preparation/manifest.json').read_text())
    for index,case in enumerate(manifest['cases']):
        seed_all(60000+index)
        states=dict(source_trained=load_state(cifar/'preparation/models/surrogate.pt'),
            victim_initial=copy.deepcopy(get_model('ConvNet','CIFAR10').state_dict()),
            victim_clean_trained=load_state(cifar/'preparation/models'/('clean-'+str(index)+'.pt')),
            victim_poison_trained=load_state(cifar/'budget-0.01-similarity/models'/('standard-'+str(index)+'.pt')))
        result['cases'].append(audit_and_align(case,load_state(cifar/'budget-0.01-similarity/poisons'/(str(case['key'])+'.pt')),
                                                states,'ConvNet','CIFAR10',data))
        write_json(root/'alignment.json',result)
        print(json.dumps(dict(event='alignment_complete',case_key=case['key'])),flush=True)
    return result


def trace(args,root,device):
    source=Path(args.inputs)
    data=data_for('MNIST',Path(args.data),device)
    manifest=json.loads((source/'mnist-original/main/manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['key']==3100000000)
    delta=load_state(source/'mnist-original/main/poisons/3100000000.pt')['delta'].to(device)
    source_state=load_state(source/'mnist-original/preparation/models/surrogate.pt')
    target=data['test_images'][case['target_id']:case['target_id']+1]
    augment=RandomTransform(28,28,shift=4,fliplr=False)
    ids=torch.tensor(case['poison_ids'],device=device)
    results=[]
    for initialization,spatial in [('fresh',True),('warm-source',True),('fresh',False)]:
        for poisoned in (False,True):
            name=initialization+('-translated' if spatial else '-no-translation')+('-poison' if poisoned else '-clean')
            path=root/'trajectories'/(name+'.json')
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            seed_all(70000)
            model=get_model('LeNet','MNIST').to(device,memory_format=torch.channels_last)
            if initialization=='warm-source':
                model.load_state_dict(source_state)
            optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=5e-4)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=[args.epochs//2.667,args.epochs//1.6,args.epochs//1.142],gamma=.1)
            images=data['images'].clone()
            if poisoned:
                images[ids]+=delta
            shuffle=torch.Generator(device=device).manual_seed(70100)
            aug_rng=torch.Generator(device=device).manual_seed(70200)
            history=[]
            start=time.perf_counter()
            for epoch in range(args.epochs):
                model.train()
                order=torch.randperm(len(data['labels']),device=device,generator=shuffle)
                for offset in range(0,len(order),128):
                    selected=order[offset:offset+128]
                    inputs,labels=images[selected],data['labels'][selected]
                    if spatial:
                        random=torch.rand(len(inputs),4,device=device,generator=aug_rng)
                        inputs=augment(inputs,random)
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(inputs),labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(name)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                model.eval()
                target_values=target_result(model,target,case['intended_label'],case['true_label'])
                correct=0
                with torch.no_grad():
                    for offset in range(0,len(data['test_labels']),512):
                        outputs=model(data['test_images'][offset:offset+512])
                        correct+=int((outputs.argmax(1)==data['test_labels'][offset:offset+512]).sum())
                entry=dict(epoch=epoch+1,clean_test_accuracy=correct/len(data['test_labels']),**target_values)
                history.append(entry)
                write_json(root/'progress.json',dict(run=name,history=history))
                print(json.dumps(dict(event='epoch',run=name,**entry)),flush=True)
            record=dict(name=name,initialization=initialization,spatial_translation=spatial,
                poisoned=poisoned,epochs=args.epochs,seed=70000,case_key=case['key'],
                training_seconds=time.perf_counter()-start,history=history,final=history[-1],
                warning='Warm start and removing translations are diagnostic interventions, not the original threat-model results.')
            write_json(path,record)
            results.append(record)
            del images,model,optimizer
            torch.cuda.empty_cache()
    write_json(root/'trajectories.json',results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',default='artifacts/local-inputs')
    parser.add_argument('--data',default='artifacts/local-data')
    parser.add_argument('--output',default='artifacts/local-attack-investigation')
    parser.add_argument('--phase',choices=['alignment','trajectory','all'],default='all')
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(root/'status.json',dict(phase='running',device=str(device)))
    try:
        if args.phase in ('alignment','all'):
            inspect(args,root,device)
        if args.phase in ('trajectory','all'):
            trace(args,root,device)
    except BaseException:
        import traceback
        write_json(root/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise
    write_json(root/'status.json',dict(phase='complete',device=str(device)))


if __name__=='__main__':
    main()
