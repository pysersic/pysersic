from numpyro import distributions as dist,infer,sample, optim
from pysersic.utils import train_numpyro_svi_early_stop
import pytest


def test_train_numpyro_svi():
    def model():
        a = sample('a', dist.Normal())

    guide =infer.autoguide.AutoDelta(model)
    svi_kernel = infer.SVI(model,guide, optim.Adam(0.001), loss = infer.Trace_ELBO())
    res = train_numpyro_svi_early_stop(svi_kernel)
    assert float(res.params['a_auto_loc']) == pytest.approx(0, abs = 1e-2)