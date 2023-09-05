from AverageMeter import AverageMeter
import torch
import time
from monai.data import DataLoader


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                loss_func: callable,
                batch_size: int,
                device: torch.device,
                max_epochs: int):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(f"Epoch {epoch+1}/{max_epochs} {idx+1}/{len(loader)} loss: {run_loss.avg} time: {time.time()-start_time}")
        start_time = time.time()
    return run_loss.avg
