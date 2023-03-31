import torch
from monai.data import DataLoader
from tensorboardX import SummaryWriter


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                optim: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                loss_fn: callable,
                device=torch.device('cuda')):
    model.train()
    run_loss = 0
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            param.grad = 0
        optim.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optim.step()
        run_loss += loss.item()


def val_epoch(model: torch.nn.Module,
              loader: DataLoader,
              loss_fn: callable):
    pass


def train_loop(model,
               train_loader,
               val_loader,
               optim,
               loss_function,
               acc_function,
               scheduler=None,
               start_epoch=0,
               max_epoch=300,
               best_acc=0,
               log_dir=None,
               post_sigmoid=None,
               post_pred=None,
               semantic_classes=None):
    writer = SummaryWriter(log_dir) if log_dir else None
    best_val_acc = best_acc
    for epoch in range(start_epoch, max_epoch):
        train_loss = train_epoch(model,
                                 train_loader,
                                 optim,
                                 epoch,
                                 loss_function)
