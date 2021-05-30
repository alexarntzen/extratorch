import torch
import torch.optim as optim
import numpy as np


def argmin(G, y_0, lr=0.1, epochs=1000, verbose=False, box_constraint=False):
    y_opt = y_0.clone().detach().requires_grad_(True)

    loss_history = []
    if box_constraint:
        optimizer = optim.Adam([y_opt], lr=lr)

        for _ in range(epochs):
            def closure():
                optimizer.zero_grad()
                g_forward = G(y_opt)
                if verbose:
                    print("loss: ", g_forward.item())
                loss_history.append(g_forward.item())
                g_forward.backward()
                return g_forward

            optimizer.step(closure=closure)
            # TODO: implement Mirror descent with Bit entropy as Bregman divergence
            # For now, only projected gradient decent is implemented
            with torch.no_grad():
                y_opt.clamp_(0, 1)

    else:
        optimizer = optim.LBFGS([y_opt], lr=float(lr), max_iter=50000, max_eval=50000, history_size=100,
                                line_search_fn="strong_wolfe", tolerance_change=1.0 * np.finfo(float).eps)

        # epochs not necessary with line search?
        def closure():
            optimizer.zero_grad()
            g_forward = G(y_opt)
            if verbose:
                print("loss: ", g_forward.item())
            loss_history.append(g_forward.item())
            g_forward.backward()
            return g_forward

        optimizer.step(closure=closure)

    return y_opt.detach()
