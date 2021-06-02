import matplotlib.pyplot as plt
import numpy as np
import torch

from deepthermal.validation import print_model_errors


def get_disc_str(model):
    params = {
        "activation": model.activation,
        "n_hidden_layers": model.n_hidden_layers,
        "neurons": model.neurons,
    }
    return str(params)


def plot_model_history(
    models,
    loss_history_trains,
    loss_history_vals=None,
    plot_name="0",
    path_figures="figures",
):
    k = len(models)
    histfig, axis = plt.subplots(1, k, figsize=(8 * k, 6))
    if k == 1:
        axis = [axis]
    for i, model in enumerate(models):
        axis[i].grid(True, which="both", ls=":")
        axis[i].loglog(
            torch.arange(1, len(loss_history_trains[i]) + 1),
            loss_history_trains[i],
            label="Training error history",
        )
        if len(loss_history_vals[i]) is not None:
            axis[i].loglog(
                torch.arange(1, len(loss_history_vals[i]) + 1),
                loss_history_vals[i],
                label="Validation error history",
            )
        axis[i].set_xlabel("Epoch")
        axis[i].set_ylabel("Loss")
        axis[i].legend()
        histfig.suptitle(f"History, model: {get_disc_str(model)}")
    histfig.savefig(f"{path_figures}/history_{plot_name}.pdf")
    plt.close(histfig)


# Todo: make this iterable over datsets
def plot_result_sorted(
    x_pred=None,
    y_pred=None,
    x_train=None,
    y_train=None,
    plot_name="vis_model",
    path_figures="../figures",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    if x_train is not None and y_train is not None:
        ax.plot(x_train, y_train, ".-.", label="training_data")
    if x_pred is not None and y_pred is not None:
        ax.plot(x_pred, y_pred, ".", label="prediction")
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# plot predicted data on
def plot_model_scatter(
    model,
    x_test,
    plot_name="vis_model",
    x_train=None,
    y_train=None,
    path_figures="../figures",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Model: {get_disc_str(model)}")
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$||x||$")
    for i in range(model.output_dimension):
        if x_train is not None and y_train is not None:
            ax.scatter(
                torch.norm(x_train, p=2, dim=1),
                y_train[:, i],
                label=f"train_{i}",
                marker=".",
            )
        ax.scatter(
            torch.norm(x_test, p=2, dim=1),
            model(x_test)[:, i].detach(),
            label=f"pred_{i}",
            marker="x",
            alpha=0.5,
            color="r",
            lw=1,
        )
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# plot predicted data on
def plot_compare_scatter(
    model, x_train, y_train, plot_name="vis_model", path_figures="../figures", **kwargs
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Model: {get_disc_str(model)}")
    ax.set_xlabel("Actual data")
    ax.set_ylabel("Predicted data")
    for i in range(model.output_dimension):
        ax.scatter(
            y_train[:, i],
            model(x_train).detach()[:, i],
            label=f"pred nr. {i}",
            marker=".",
            lw=1,
        )
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# plot visualization
def plot_model_1d(
    model,
    x_test,
    plot_name="vis_model",
    x_train=None,
    y_train=None,
    path_figures="../figures",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Model: {get_disc_str(model)}")
    for i in range(model.output_dimension):
        if x_train is not None and y_train is not None:
            ax.scatter(x_train[:, 0], y_train[:, i], label=f"train_{i}")
        ax.plot(
            x_test[:, 0], model(x_test)[:, i].detach(), label=f"pred_{i}", lw=2, ls="-."
        )
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


def plot_result(
    models,
    loss_history_trains,
    loss_history_vals,
    rel_val_errors,
    path_figures,
    plot_name,
    plot_function,
    function_kwargs,
    model_list=None,
    history=True,
    **kwargs,
):
    if model_list is None:
        model_list = np.arange(len(models))
    print_model_errors(rel_val_errors)
    for i in model_list:
        if history:
            plot_model_history(
                models[i],
                loss_history_trains[i],
                loss_history_vals[i],
                plot_name=f"{plot_name}_{i}",
                path_figures=path_figures,
            )
        for j in range(len(models[i])):
            plot_function(
                plot_name=f"{plot_name}_{i}_{j}",
                model=models[i][j],
                path_figures=path_figures,
                **function_kwargs,
            )
