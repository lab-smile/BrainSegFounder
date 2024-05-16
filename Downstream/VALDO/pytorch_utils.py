import torch

def distributed_torch_available():
    if not torch.distributed.is_available():
        raise RuntimeError("Requires distributed package to be available")


