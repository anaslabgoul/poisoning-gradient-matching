"""Inference-only evaluation of saved victims on an entire CIFAR-10 class.

Reports the fraction of source-class images predicted as the intended class,
and the same rate on matched clean controls. The current poisons were designed
for ONE specific image; this script measures their class-wide transfer, not a
new class-targeted attack. No model is trained or modified.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forest.consts import cifar10_mean, cifar10_std
from forest.victims.models import get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='experiments/results/four-model-comparison.json')
    parser.add_argument('--data', default='~/data')
    parser.add_argument('--output', default='artifacts/four-model-comparison/dog-group-evaluation.json')
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding='utf-8-sig'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this saved-model evaluation.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    dataset = torchvision.datasets.CIFAR10(str(Path(args.data).expanduser()), train=False, download=False)
    labels = torch.tensor(dataset.targets, device=device)
    dm = torch.tensor(cifar10_mean, device=device)[None, :, None, None]
    ds = torch.tensor(cifar10_std, device=device)[None, :, None, None]
    images = torch.tensor(dataset.data, device=device).permute(0, 3, 1, 2).float()/255
    images = ((images-dm)/ds).contiguous(memory_format=torch.channels_last)
    source_label, intended_label = 5, 6
    target_id = reference['target_id']
    group = labels == source_label
    unseen = group.clone()
    unseen[target_id] = False
    assert int(group.sum()) == 1000 and int(unseen.sum()) == 999
    bf16 = reference['precision'].startswith('BF16')

    def evaluate(checkpoint):
        model = get_model('ResNet18', 'CIFAR10').to(device, memory_format=torch.channels_last)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        predictions = torch.empty(len(labels), dtype=torch.long, device=device)
        probabilities = torch.empty(len(labels), device=device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(labels), 256):
                context = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext()
                with context:
                    output = model(images[start:start+256]).float()
                predictions[start:start+256] = output.argmax(1)
                probabilities[start:start+256] = output.softmax(1)[:, intended_label]
        torch.cuda.synchronize()
        group_count = int(((predictions == intended_label) & group).sum())
        unseen_count = int(((predictions == intended_label) & unseen).sum())
        clean_correct = int((predictions == labels).sum())
        result = dict(checkpoint=str(checkpoint), clean_test_correct=clean_correct,
                      clean_test_total=len(labels), clean_test_accuracy_pct=100*clean_correct/len(labels),
                      dog_to_frog_count=group_count, dog_total=int(group.sum()), dog_to_frog_rate_pct=100*group_count/int(group.sum()),
                      unseen_dog_to_frog_count=unseen_count, unseen_dog_total=int(unseen.sum()),
                      unseen_dog_to_frog_rate_pct=100*unseen_count/int(unseen.sum()),
                      dog_correct=int(((predictions == source_label) & group).sum()),
                      dog_prediction_counts=torch.bincount(predictions[group], minlength=10).tolist(),
                      mean_frog_probability_on_dogs=float(probabilities[group].mean()),
                      crafted_target_prediction=int(predictions[target_id]),
                      inference_seconds=time.perf_counter()-started)
        del model
        torch.cuda.empty_cache()
        return result

    rows = []
    for row in reference['results']:
        checkpoint = Path(row['checkpoint'])
        attacked = evaluate(checkpoint)
        control_path = checkpoint.parent/('clean-1729-'+row['method']+'.pt')
        control = evaluate(control_path)
        record = dict(method=row['method'], poisoned_model=attacked, clean_control=control,
                      group_rate_change_pp=attacked['dog_to_frog_rate_pct']-control['dog_to_frog_rate_pct'])
        rows.append(record)
        print(json.dumps(dict(method=row['method'],
            clean_test_accuracy_pct=attacked['clean_test_accuracy_pct'],
            dog_to_frog_count=attacked['dog_to_frog_count'], dog_total=attacked['dog_total'],
            dog_to_frog_rate_pct=attacked['dog_to_frog_rate_pct'],
            control_dog_to_frog_rate_pct=control['dog_to_frog_rate_pct'])), flush=True)
    result = dict(evaluation='Class-wide transfer of the existing single-image poisons; inference only',
                  source_group='All CIFAR-10 test dogs', intended_class='frog',
                  group_size=1000, crafted_target_id=target_id, unseen_group_size=999,
                  clean_test_size=10000, classes=dataset.classes, training_source_commit=reference['source_commit'],
                  limitation='Only one dog image was used in poison construction. These are not results of a class-targeted poison optimization.',
                  results=rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
