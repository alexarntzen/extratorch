import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from typing import Union, List
from collections.abc import Callable

# GLOBAL VARIABLES
LRS = optim.lr_scheduler._LRScheduler
InitScheduler = Callable[[optim.Optimizer], LRS]
larning_rates = {"ADAM": 0.001, "LBFGS": 0.1, "strong_wolfe": 1}


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if "weight" or "bias" in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def compute_loss_torch(
    model: nn.Module, data: Union[List, Dataset], loss_func: callable
) -> torch.Tensor:
    """default way"""
    x_train, y_train = data[:]
    y_pred = model(x_train)
    loss = loss_func(y_pred, y_train)
    return loss


def fit_module(
    model: nn.Module,
    data,
    num_epochs,
    batch_size,
    optimizer,
    init: callable = None,
    regularization_param=0,
    regularization_exp=2,
    data_val=None,
    track_history=True,
    track_epoch=True,
    verbose=False,
    learning_rate=None,
    init_weight_seed: int = None,
    lr_scheduler: InitScheduler = None,
    loss_func=nn.MSELoss(),
    compute_loss: Callable[..., torch.Tensor] = compute_loss_torch,
    max_nan_steps=50,
    post_batch: Callable = None,
    post_epoch: Callable = None,
    **kwargs,
) -> tuple[nn.Module, pd.DataFrame]:
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

    if learning_rate is None and not callable(optimizer):
        learning_rate = larning_rates[optimizer]
    # select optimizer
    if optimizer == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer_ = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "LBFGS":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            max_iter=1,
            max_eval=50000,
            tolerance_change=1.0 * np.finfo(float).eps,
            lr=learning_rate,
        )
    elif optimizer == "strong_wolfe":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=100,
            max_eval=1000,
            history_size=200,
            line_search_fn="strong_wolfe",
        )
        max_nan_steps = 2
    elif callable(optimizer):
        optimizer_ = optimizer(model.parameters())
    else:
        raise ValueError("Optimizer not recognized")

    # Learning Rate Scheduler

    scheduler = lr_scheduler(optimizer_) if lr_scheduler is not None else None

    loss_history_train = list()
    loss_history_val = list()
    if track_epoch:
        loss_history_train = np.zeros(num_epochs)
        loss_history_val = np.zeros(num_epochs)

    nan_steps = 0
    # Loop over epochs
    epochs_tqdm = tqdm(
        range(num_epochs), desc="Epoch: ", disable=(not verbose), leave=False
    )
    for epoch in epochs_tqdm:
        try:
            # try one epoch, break if interupted:
            training_set = DataLoader(
                data, batch_size=batch_size, shuffle=True, drop_last=False
            )
            for j, data_sample in enumerate(training_set):

                def closure():
                    # zero the parameter gradients
                    optimizer_.zero_grad()
                    # forward + backward + optimize
                    loss_u = compute_loss(
                        model=model,
                        data=data_sample,
                        loss_func=loss_func,
                    )
                    loss_reg = regularization(model, regularization_exp)
                    loss = loss_u + regularization_param * loss_reg
                    loss.backward()

                    return loss

                optimizer_.step(closure=closure)

                if post_batch is not None:
                    post_batch(model=model, data=data)

                # track after each step if not track epoch
                # assumes that the expected loss is
                # not proportional to the length of training data
                if track_history and not track_epoch:
                    # track training loss
                    train_loss = compute_loss(
                        model=model, data=data, loss_func=loss_func
                    ).item()
                    loss_history_train.append(train_loss)

                    # track validation loss
                    if data_val is not None and len(data_val) > 0 and track_history:
                        validation_loss = compute_loss(
                            model=model, data=data_val, loss_func=loss_func
                        ).item()
                        loss_history_val.append(validation_loss)

            if post_epoch is not None:
                post_epoch(model=model, data=data)

            if track_epoch or track_history or lr_scheduler:
                train_loss = compute_loss(
                    model=model, data=data, loss_func=loss_func
                ).item()
                if track_history:
                    # stop if nan output
                    if np.isnan(train_loss):
                        nan_steps += 1
                    if epoch % 100 == 0:
                        nan_steps = 0

                if lr_scheduler is not None:
                    scheduler.step(train_loss)

                if track_epoch:
                    loss_history_train[epoch] = train_loss
                    if data_val is not None and len(data_val) > 0:
                        validation_loss = compute_loss(
                            model=model,
                            data=data_val,
                            loss_func=loss_func,
                        ).item()
                        loss_history_val[epoch] = validation_loss
            if verbose and track_history:
                print_iter = epoch if track_epoch else -1
                if data_val is not None and len(data_val) > 0:
                    epochs_tqdm.set_postfix(
                        loss=loss_history_train[print_iter],
                        val_loss=loss_history_val[print_iter],
                    )
                else:
                    epochs_tqdm.set_postfix(
                        loss=loss_history_train[print_iter],
                    )
            if nan_steps > max_nan_steps:
                break

        except KeyboardInterrupt:
            print("Interrupted breaking")
            break

    if verbose and track_history and len(loss_history_train) > 0:
        print("Final training loss: ", np.round(loss_history_train[-1], 8))
        if data_val is not None and len(data_val) > 0:
            print("Final validation Loss: ", np.round(loss_history_val[-1], 8))

    history = pd.DataFrame(
        {"Validation loss": loss_history_train, "Training loss": loss_history_val}
    )
    history.index.name = "Epochs" if track_epoch else "Iterations"
    return model, history
