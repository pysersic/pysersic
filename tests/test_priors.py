import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed

from pysersic.priors import (
    FlatSkyPrior,
    NoSkyPrior,
    PySersicMultiPrior,
    PySersicSourcePrior,
    SourceProperties,
    TiltedPlaneSkyPrior,
    estimate_sky,
)

prof_names = ["sersic", "doublesersic", "pointsource", "exp", "dev"]
prof_vars = [
    ["xc", "yc", "flux", "r_eff", "n", "ellip", "theta"],
    [
        "xc",
        "yc",
        "flux",
        "f_1",
        "r_eff_1",
        "n_1",
        "ellip_1",
        "r_eff_2",
        "n_2",
        "ellip_2",
        "theta",
    ],
    ["xc", "yc", "flux"],
    ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
    ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
]


@pytest.mark.parametrize("mask", [None, jnp.zeros((100, 100))])
def test_SourceProperties(mask):
    test_im = jnp.ones((100, 100))
    SP = SourceProperties(test_im, mask=mask)
    SP.measure_properties()
    attributes = [
        "sky_guess",
        "sky_guess_err",
        "flux_guess",
        "flux_guess_err",
        "r_eff_guess",
        "r_eff_guess_err",
        "xc_guess",
        "yc_guess",
    ]
    for var in attributes:
        assert hasattr(SP, var)


@pytest.mark.parametrize("prof, var_names", zip(prof_names, prof_vars))
def test_prior_gen(prof, var_names):
    props = SourceProperties(jnp.ones((100, 100)))
    prior_class = props.generate_prior(profile_type=prof)
    assert prior_class.check_vars(verbose=True)


def test_sky_sampling():
    x = jnp.arange(100)
    X, Y = jnp.meshgrid(x, x)

    prior_class = NoSkyPrior(sky_guess=0, sky_guess_err=1.0)
    with seed(rng_seed=1):
        params_1 = prior_class.sample(X, Y)
    assert params_1 == 0

    prior_class = FlatSkyPrior(sky_guess=0, sky_guess_err=1)
    with seed(rng_seed=1):
        params_2 = prior_class.sample(X, Y)
    assert params_2.shape == ()  # Should be single value

    prior_class = TiltedPlaneSkyPrior(sky_guess=0, sky_guess_err=1)
    with seed(rng_seed=1):
        params_3 = prior_class.sample(X, Y)
    assert params_3.shape == (100, 100)  # Should be 2D array


def test_PySersicSourcePrior():
    prior = PySersicSourcePrior("pointsource")
    prior.set_gaussian_prior("xc", 10.0, 1.0)
    prior.set_gaussian_prior("yc", 10.0, 1.0)
    prior.set_uniform_prior("flux", 0, 100)
    with seed(rng_seed=1):
        params = prior()
    assert params["xc"] == pytest.approx(9.7560796737, rel=1e-5)
    assert params["yc"] == pytest.approx(9.7560796737, rel=1e-5)
    assert params["flux"] == pytest.approx(37.12473, rel=1e-5)


def test_PySersicMultiPrior():
    catalog = {}
    catalog["x"] = [10, 15, 20]
    catalog["y"] = [20, 15, 10]
    catalog["flux"] = [100, 100, 100]
    catalog["r"] = [5, 5, 5]
    catalog["theta"] = [0, 0, 0]
    catalog["type"] = ["pointsource", "exp", "sersic"]
    mp = PySersicMultiPrior(catalog)
    with seed(rng_seed=1):
        params = mp()

    assert params["flux_0"] == pytest.approx(95.121597, abs=1e-4)
    assert params["xc_0"] == pytest.approx(8.90784, abs=1e-4)
    assert params["yc_0"] == pytest.approx(19.67145, abs=1e-4)
    assert params["flux_1"] == pytest.approx(113.12357, abs=1e-4)
    assert params["xc_1"] == pytest.approx(13.87507, abs=1e-4)
    assert params["yc_1"] == pytest.approx(16.05198, abs=1e-4)
    assert params["r_eff_1"] == pytest.approx(9.10913, abs=1e-4)
    assert params["ellip_1"] == pytest.approx(0.32804, abs=1e-4)
    assert params["theta_1"] == pytest.approx(6.21555, abs=1e-4)
    assert params["flux_2"] == pytest.approx(65.84653, abs=1e-4)
    assert params["xc_2"] == pytest.approx(21.38217, abs=1e-4)
    assert params["yc_2"] == pytest.approx(11.08138, abs=1e-4)
    assert params["r_eff_2"] == pytest.approx(3.35782, abs=1e-4)
    assert params["ellip_2"] == pytest.approx(0.78032, abs=1e-4)
    assert params["theta_2"] == pytest.approx(0.61594, abs=1e-4)
    assert params["n_2"] == pytest.approx(4.79926, abs=1e-4)


def test_sky_estimate():
    rng = np.random.default_rng(1234567)
    test_im = rng.normal(size=(100, 100))
    med, std, npix = estimate_sky(test_im, n_pix_sample=5)
    assert med == pytest.approx(0.0, abs=1e-2)
    assert std == pytest.approx(1.0, abs=1e-2)
    assert npix == 1900
