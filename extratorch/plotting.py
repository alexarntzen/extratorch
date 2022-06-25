import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import numpy as np
import torch
import warnings
from extratorch.validation import print_model_errors
import os
from typing import List, Union


def get_disc_str(model):
    params = {
        "activation": model.activation,
        "n_hidden_layers": model.n_hidden_layers,
        "neurons": model.neurons,
    }
    return str(params)


def plot_model_history(
    histories: Union[List[pd.DataFrame], pd.DataFrame],
    plot_name="Loss history",
    path_figures="figures",
):

    if not isinstance(histories, List):
        return plot_model_history(
            [histories], plot_name=plot_name, path_figures=path_figures
        )

    if len(histories) == 0:
        warnings.warn("No loss history to plot")
        return

    k = len(histories)
    histfig, axis = plt.subplots(1, k, tight_layout=True)

    if k == 1:
        axis = [axis]
    for i, history in enumerate(histories):
        x_values = history.index
        for col in history:
            y_values = history[col].values
            if not len(y_values) > 0:
                continue
            if np.any(y_values < 0) or (
                y_values is not None and np.any(y_values[i] < 0)
            ):
                plot_func = axis[i].semilogx
            else:
                plot_func = axis[i].loglog

            plot_func(x_values, y_values, label=col)

        axis[i].set_xlabel(history.index.name)
        axis[i].legend()

    histfig.savefig(os.path.join(path_figures, f"history_{plot_name}.pdf"))
    plt.close(histfig)


# Todo: make this iterable over datsets
def plot_result_sorted(
    x_pred=None,
    y_pred=None,
    x_train=None,
    y_train=None,
    plot_name="",
    x_axis="",
    y_axis="",
    path_figures="../figures",
    compare_label="Data",
    pred_label="Prediction",
    fig: Figure = None,
    ax=None,
    save=True,
    color_pred=None,
    color_train=None,
) -> Figure:
    if ax is not None:
        fig = ax.get_figure()
    elif fig is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        ax = fig.get_axes()[0]

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if x_train is not None and y_train is not None:
        ax.plot(
            x_train, y_train, ".:", label=compare_label, color=color_train, lw=2, mew=1
        )
    if x_pred is not None and y_pred is not None:
        ax.plot(x_pred, y_pred, "-", label=pred_label, color=color_pred, lw=2)
    ax.legend()
    if save:
        fig.savefig(os.path.join(path_figures, f"{plot_name}.pdf"))
        plt.close(fig)
    return fig


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
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$||x||$")
    for i in range(y_train.size(-1)):
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
    fig.savefig(os.path.join(path_figures, f"{plot_name}.pdf"))
    plt.close(fig)


# plot predicted data on
def plot_compare_scatter(
    model, x_train, y_train, plot_name="vis_model", path_figures="../figures", **kwargs
):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel("Actual data")
    ax.set_ylabel("Predicted data")
    for i in range(y_train.size(-1)):
        ax.scatter(
            y_train[:, i],
            model(x_train).detach()[:, i],
            label=f"pred nr. {i}",
            marker=".",
            lw=1,
        )
    ax.legend()
    fig.savefig(os.path.join(path_figures, f"{plot_name}.pdf"))
    plt.close(fig)


# plot visualization
def plot_model_1d(model, x_test, **kwargs):
    y_pred = model(x_test).detach()
    plot_result_sorted(y_pred=y_pred, x_pred=x_test, **kwargs)


def plot_result(
    models,
    histories: List[pd.DataFrame] = None,
    path_figures="",
    plot_name="plot",
    rel_val_errors=None,
    plot_function=None,
    function_kwargs=None,
    model_list=None,
    model_params: pd.DataFrame = None,
    training_params: pd.DataFrame = None,
    **kwargs,
):
    if model_list is None:
        model_list = np.arange(len(models))
    if rel_val_errors is not None:
        print_model_errors(rel_val_errors)
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)
    if model_params is not None:
        model_params.to_csv(os.path.join(path_figures, "model_params.csv"))
    if training_params is not None:
        training_params.to_csv(os.path.join(path_figures, "training_params.csv"))
    for i in model_list:
        t, c, f = training_params.loc[i][["trial", "config", "fold"]]
        if histories is not None:
            plot_model_history(
                histories[i],
                plot_name=f"{plot_name}_{i}",
                path_figures=path_figures,
            )
            histories[i].to_csv(
                os.path.join(path_figures, f"history_{plot_name}_t{t}_c{c}_f{f}.csv")
            )
        if plot_function is not None:
            plot_function(
                plot_name=f"_t{t}_c{c}_f{f}",
                model=models[i],
                path_figures=path_figures,
                **function_kwargs,
                **kwargs,
            )
