"""GPU smoke test and throughput estimate, without downloading any data."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forest.victims.bilevel import bilevel_backward
from forest.victims.models import get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--train-images', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--radius', type=float, default=16.)
    parser.add_argument('--classification-bf16', action='store_true')
    parser.add_argument('--regularizer-samples', type=int, default=None)
    parser.add_argument('--output', default='artifacts/bilevel-resnet18/benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = get_model(args.model, 'CIFAR10').cuda()
    if args.classification_bf16:
        model = model.to(memory_format=torch.channels_last)
    x = torch.rand(args.batch, 3, 32, 32, device='cuda')
    y = torch.arange(args.batch, device='cuda') % 10
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    records = []
    for iteration in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, metrics = bilevel_backward(model, x, y, .0001, radius=args.radius,
            steps=args.steps, chunk_size=args.chunk, classification_bf16=args.classification_bf16,
            regularizer_samples=args.regularizer_samples)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('GPU full-network outer gradient is invalid.')
        record = dict(iteration=iteration, seconds=elapsed, batch=args.batch, chunk=args.chunk,
            model=args.model, steps=args.steps, regularizer_samples=args.regularizer_samples,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            gradient_parameter_tensors=len(list(model.parameters())),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            metrics={k: float(v) for k, v in metrics.items()},
            projected_training_hours=elapsed/args.batch*args.train_images*args.epochs/3600,
            training_images=args.train_images, epochs=args.epochs)
        records.append(record)
        print(json.dumps(record), flush=True)
        optimizer.step()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
