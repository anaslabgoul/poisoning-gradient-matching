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
    parser.add_argument('--output', default='artifacts/bilevel-resnet18/benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = get_model('ResNet18', 'CIFAR10').cuda()
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
        _, metrics = bilevel_backward(model, x, y, .0001, steps=args.steps, chunk_size=args.chunk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('GPU full-network outer gradient is invalid.')
        record = dict(iteration=iteration, seconds=elapsed, batch=args.batch, chunk=args.chunk,
            steps=args.steps, trainable_parameters=sum(p.numel() for p in model.parameters()),
            gradient_parameter_tensors=len(list(model.parameters())),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            metrics={k: float(v) for k, v in metrics.items()},
            projected_hours_per_40_epoch_model=elapsed/args.batch*50000*40/3600)
        records.append(record)
        print(json.dumps(record), flush=True)
        optimizer.step()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
