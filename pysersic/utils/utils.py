import jax.numpy as jnp
from typing import Union, Optional, Callable
from numpyro.infer.svi import SVI, SVIRunResult
from numpyro import optim
from jax.random import PRNGKey
from jax import jit
from functools import partial
import copy
import tqdm


def train_numpyro_svi_early_stop(
        svi_class: SVI,
        num_round: Optional[int] = 3,
        max_train: Optional[int] = 5000,
        lr_init: Optional[float] = 0.01,
        frac_lr_decrease: Optional[float]  = 0.1,
        patience: Optional[int] = 100,
        optimizer: Optional[optim._NumPyroOptim] = optim.Adam,
        rkey: Optional[PRNGKey] = PRNGKey(10),
    )-> SVIRunResult:
    """Optimize a SVI model by training for multiple rounds with a deacreasing learning rate, and early stopping for each round

    Parameters
    ----------
    svi_class : SVI
        Initialized numpyo SVI class, note that the optimizer will be overwritten
    num_round : Optional[int], optional
        Number of training rounds, by default 3
    max_train : Optional[int], optional
        Max number of training epochs per ropund, by default 3000
    lr_init : Optional[float], optional
        Initial learning rate, by default 0.1
    frac_lr_decrease : Optional[float], optional
        Multiplicative factor to change learning rate each round, by default 0.1
    patience : Optional[int], optional
        Number of training epochs to wait for improvement, by default 100
    optimizer : Optional[optim._NumPyroOptim], optional
        Optimizer algorithm tro use, by default optim.Adam
    rkey : Optional[PRNGKey], optional
        Jax PRNG key, by default PRNGKey(10)

    Returns
    -------
    SVIRunResult
        SVI Result class containing trained model
    """
    optim_init = optimizer(lr_init)
    svi_class.__setattr__('optim', optim_init)

    init_state = svi_class.init(rkey)
    all_losses = []

    @partial(jit, static_argnums = 1)
    def update_func(state,svi_class,lr):
        svi_class.__setattr__('optim', optimizer(lr))
        state,loss = svi_class.stable_update(state)
        return state,loss

    best_state, best_loss = update_func(init_state, svi_class, lr_init)
    
    for r in range(num_round):
        losses = []
        wait_counter = 0
        svi_state = copy.copy(best_state)

        if r>0:
            lr_cur = lr_init*frac_lr_decrease**r
            best_loss = jnp.inf
        else:
            lr_cur = lr_init

        with tqdm.trange(1, max_train + 1) as t:
            for j in t:
                svi_state, loss = update_func(svi_state, svi_class, lr_cur)
                if loss < best_loss:
                    best_loss = loss
                    best_state = copy.copy(svi_state)
                    wait_counter = 0
                elif wait_counter >= patience:
                    break
                else:
                    wait_counter += 1
                t.set_postfix_str(f'Round = {r:d},step_size = {lr_cur:.1e} loss: {best_loss:.3e}',refresh=False)
                losses.append(loss)
        
        all_losses.append(losses)

    return SVIRunResult(svi_class.get_params(best_state), svi_state,losses)
    
