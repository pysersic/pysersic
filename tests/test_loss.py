import pytest
import jax.numpy as jnp
from pysersic import loss
from numpyro.handlers import seed

loss_func_list = [loss.gaussian_loss, loss.cash_loss,loss.gaussian_loss_w_frac,
                loss.gaussian_loss_w_sys, loss.student_t_loss, loss.student_t_loss_free_sys,
                loss.pseudo_huber_loss, loss.gaussian_mixture, loss.gaussian_mixture_w_sys, loss.gaussian_mixture_w_frac
]


data = jnp.ones((100,100))
sig = jnp.ones((100,100))
rms = jnp.ones((100,100))
mask = jnp.ones((100,100), dtype=bool)

@pytest.mark.parametrize("loss_func", loss_func_list)
def test_all_losses(loss_func):
    with seed(rng_seed=0):
        loss_val = loss_func(data,sig,rms,mask)