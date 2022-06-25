import copy
import warnings

import numpy as np
import torch
import torch.utils
from torch.utils.data import Subset, DataLoader, Dataset
import pandas as pd
import itertools
from tqdm.auto import tqdm
from itertools import product as prod
from sklearn.model_selection import KFold
from collections.abc import Callable
from extratorch.train import fit_module
from extratorch.models import get_scaled_model


# Root Relative Squared Error
def get_RRSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    y_data_mean = torch.mean(y_data, dim=0)
    relative_error_2 = torch.sum((y_pred - y_data) ** 2) / torch.sum(
        (y_data_mean - y_data) ** 2
    )
    relative_error = relative_error_2**0.5
    if verbose:
        print(
            f"Root Relative Squared {type_str} Error: ",
            relative_error.item() * 100,
            "%",
        )
    return relative_error.item()


# Normalized root-mean-square error
def get_NRMSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    error = torch.mean((y_pred - y_data) ** 2) ** 0.5
    if verbose:
        print(f"Root mean square {type_str} error: ", error.item() * 100, "%")
    return error.item()


def k_fold_cv_grid(
    model_params,
    training_params,
    data: Dataset = None,
    shuffle_folds: bool = True,
    val_data: Dataset = None,
    fit: Callable[..., (torch.nn.Module, pd.DataFrame)] = fit_module,
    folds=1,
    trials=1,
    partial=False,
    verbose=True,
    print_params: bool = False,
    get_error=None,
    copy_data: bool = False,
):
    # transform a dictionary with a list of inputs
    # into an iterator over the coproduct of the lists
    if isinstance(model_params, dict):
        model_params_iter = create_subdictionary_iterator(model_params)
    else:
        model_params_iter = model_params

    if isinstance(training_params, dict):
        training_params_iter = create_subdictionary_iterator(training_params)
    else:
        training_params_iter = training_params

    models = []
    histories = []
    rel_train_errors = []
    rel_val_errors = []
    model_params_dfs = []
    training_params_dfs = []
    models_iter_tqdm = tqdm(
        enumerate(
            prod(
                range(trials),
                enumerate(prod(model_params_iter, training_params_iter)),
                enumerate(
                    _get_data_splits(
                        folds=folds, data=data, val_data=val_data, shuffle=shuffle_folds
                    )
                ),
            )
        ),
        disable=(not verbose),
        leave=False,
    )
    for index, (trial, (config, config_params), (fold, fold_data)) in models_iter_tqdm:

        # expand params and data
        fold_train_data, fold_val_data = fold_data
        model_param, training_param = config_params

        # print and store running information
        run_dict = dict(config=config, trial=trial, fold=fold)

        model_params_df = pd.DataFrame(
            (model_param | run_dict,), index=[index], dtype=object
        )
        training_params_df = pd.DataFrame(
            (training_param | run_dict,), index=[index], dtype=object
        )
        if verbose:
            models_iter_tqdm.set_postfix(trial=trial, config=config, fold=fold)
            if print_params:
                print("Model params:")
                pretty_dict_print(model_param)
                print("Training params:")
                pretty_dict_print(training_param)

        # train model on data!
        model_param_k = model_param.copy()
        model_instance = model_param_k.pop("model")(**model_param_k)
        model_instance, history = fit(
            model=model_instance,
            **training_param,
            data=copy.deepcopy(fold_train_data) if copy_data else fold_train_data,
            data_val=copy.deepcopy(fold_val_data) if copy_data else fold_val_data,
        )

        model_params_dfs.append(model_params_df)
        training_params_dfs.append(training_params_df)

        models.append(model_instance)
        histories.append(history)
        if callable(get_error):
            rel_train_errors.append(get_error(model_instance, fold_train_data))
            rel_val_errors.append(get_error(model_instance, fold_val_data))

        if partial:
            break

    k_fold_grid = {
        "models": models,
        "model_params": pd.concat(model_params_dfs),
        "training_params": pd.concat(training_params_dfs),
        "histories": histories,
        "rel_train_errors": rel_train_errors,
        "rel_val_errors": rel_val_errors,
    }
    return k_fold_grid


def _get_data_splits(folds=1, data=None, val_data=None, shuffle=True):
    # do some validation here so to avoid mistakes
    if data is None:
        assert folds == 1, "will not do multiple folds no data"
    if val_data is not None and folds != 1:
        warnings.warn("Got validation data, will not split training data")
        folds = 1

    if folds == 1:
        yield data, val_data
    else:
        for train_index, val_index in KFold(n_splits=folds, shuffle=shuffle).split(
            data
        ):
            yield Subset(data, train_index), Subset(data, val_index)


# like Tor
def item_to_list(*values, cycle=False):
    for value in values:
        if not isinstance(value, (list, np.ndarray)):
            value = itertools.cycle([value]) if cycle else [value]
        yield value


def create_subdictionary_iterator(dictionary: dict, product=True) -> iter:
    """Create an iterator over a dictionary of lists
    Important: all lists in the dict must be of the same lenght if zip is chosen
    Cartesian product is default
    """
    combine = itertools.product if product else zip

    for sublist in combine(*item_to_list(*dictionary.values(), cycle=not product)):
        # convert two list into dictionary

        yield dict(zip(dictionary.keys(), sublist))


def add_dictionary_iterators(*dict_iterators: iter, product=True):
    """Combine two subdictionary iterators"""
    combine = itertools.product if product else zip
    for left, right in combine(*dict_iterators):
        yield right | left


# printing model errors
def print_model_errors(rel_val_errors, **kwargs):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} average error: {avg_error}")


def pretty_dict_print(d: dict):
    for key, value in d.items():
        print(f"{key}: {value}", end=", ")
    print("")


def get_scaled_results(cv_results, x_center=0, x_scale=1, y_center=0, y_scale=1):
    cv_results_scaled = cv_results.copy()
    cv_results_scaled["models"] = []
    for i in range(len(cv_results["models"])):
        cv_results_scaled["models"][i].append(
            get_scaled_model(
                cv_results["models"][i],
                x_center=x_center,
                x_scale=x_scale,
                y_center=y_center,
                y_scale=y_scale,
            )
        )
    return cv_results_scaled
