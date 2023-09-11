import datetime
import os
import matplotlib.pyplot as plt


def visualize(losses: list, dices: tuple, epochs, out_dir='./figures'):
    train_dices, val_dices = dices
    train_tcs, train_wts, train_ets, train_avgs = train_dices
    val_tcs, val_wts, val_ets, val_avgs = val_dices

    os.makedirs(out_dir, exist_ok=True)
    loss_curve(epochs, losses, title='Training Loss', out_dir=out_dir)

    for title, dices in zip(['Val Dice TC', 'Val Dice WT', 'Val Dice ET', 'Val Average Dice',
                             'Train Dice TC', 'Train Dice WT', 'Train Dice ET', 'Train Average Dice'],
                            [val_tcs, val_wts, val_ets, val_avgs, train_tcs, train_wts, train_ets, train_avgs]):
        dice_curve(epochs, dices, title=title, out_dir=out_dir)


def loss_curve(epochs: list, losses: list, title: str, out_dir: str) -> None:
    plt.plot(epochs, losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (1 - Average Dice)')
    plt.savefig(f'{out_dir}/{title}_{str(datetime.datetime.now())}.svg')


def dice_curve(epochs: list, dices: list, title: str, out_dir: str) -> None:
    plt.plot(epochs, dices)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.savefig(f'{out_dir}/{title}_{str(datetime.datetime.now())}.svg')


