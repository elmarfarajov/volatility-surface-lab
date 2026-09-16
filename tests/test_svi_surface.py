import numpy as np
import pytest

from volsurf.surface import VolSurface
from volsurf.svi import SVISlice, fit_svi_slice

GOOD = SVISlice(a=0.01, b=0.08, rho=-0.6, m=0.02, s=0.15, maturity=0.5)


def test_fit_recovers_a_known_slice():
    k = np.linspace(-0.5, 0.3, 35)
    fit = fit_svi_slice(k, GOOD.implied_vol(k), GOOD.maturity)
    assert fit.rmse_vol < 2e-4
    np.testing.assert_allclose(fit.slice.implied_vol(k), GOOD.implied_vol(k), atol=5e-4)


def test_fitted_slice_is_arbitrage_free_and_respects_lee_bound():
    rng = np.random.default_rng(0)
    k = np.linspace(-0.6, 0.3, 40)
    noisy = GOOD.implied_vol(k) + 0.003 * rng.standard_normal(k.size)
    fit = fit_svi_slice(k, noisy, GOOD.maturity)
    assert fit.min_durrleman_g >= -1e-6
    assert fit.slice.lee_wing_slope <= 2.0 + 1e-9
    assert fit.slice.min_variance >= -1e-9


def test_calendar_constraint_against_previous_slice():
    k = np.linspace(-0.4, 0.3, 30)
    longer_maturity = 0.6
    crossing_quotes = np.sqrt(GOOD.total_variance(k) * 0.9 / longer_maturity)
    fit = fit_svi_slice(k, crossing_quotes, longer_maturity, previous=GOOD)
    assert fit.calendar_violation < 1e-4


def test_detects_the_classic_butterfly_arbitrage_example():
    vogt = SVISlice(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153, maturity=1.0)
    assert vogt.durrleman_g(np.linspace(-1.5, 1.5, 601)).min() < 0


def test_density_integrates_to_one():
    k = np.linspace(-4, 4, 40_001)
    total = np.trapezoid(GOOD.risk_neutral_density(k), k)
    assert total == pytest.approx(1.0, abs=2e-3)


def test_too_few_quotes_raises():
    with pytest.raises(ValueError):
        fit_svi_slice([0.0, 0.1], [0.2, 0.2], 0.5)


def _flat_surface(vol: float) -> VolSurface:
    maturities = [0.25, 0.5, 1.0, 2.0]
    slices = [SVISlice(a=vol**2 * t, b=0.0, rho=0.0, m=0.0, s=0.1, maturity=t) for t in maturities]
    return VolSurface(slices, [100.0] * 4, [1.0] * 4, 100.0)


def test_flat_surface_has_flat_local_vol():
    surface = _flat_surface(0.2)
    k = np.linspace(-0.3, 0.3, 7)
    for t in (0.1, 0.3, 0.75, 1.5):
        np.testing.assert_allclose(surface.implied_vol(k, t), 0.2, atol=1e-10)
        np.testing.assert_allclose(surface.local_vol(k, t), 0.2, atol=1e-6)


def test_surface_reproduces_slices_and_is_calendar_monotone():
    slices = [
        SVISlice(a=0.002, b=0.05, rho=-0.7, m=0.0, s=0.05, maturity=0.1),
        SVISlice(a=0.01, b=0.08, rho=-0.65, m=0.01, s=0.12, maturity=0.5),
        SVISlice(a=0.025, b=0.1, rho=-0.6, m=0.02, s=0.2, maturity=1.0),
    ]
    surface = VolSurface(slices, [100.2, 101.0, 102.0], [0.996, 0.98, 0.96], 100.0)
    k = np.linspace(-0.4, 0.3, 15)
    for sl in slices:
        np.testing.assert_allclose(surface.total_variance(k, sl.maturity), sl.total_variance(k), atol=1e-12)
    grid_t = np.linspace(0.05, 1.5, 40)
    w = np.array([surface.total_variance(k, t) for t in grid_t])
    assert (np.diff(w, axis=0) >= -1e-12).all()
    assert surface.calendar_violations(k).max() == 0.0


def test_forward_interpolation_is_log_linear():
    surface = _flat_surface(0.2)
    surface.forwards = np.array([101.0, 102.0, 104.0, 108.0])
    assert surface.forward(0.25) == pytest.approx(101.0)
    assert surface.forward(0.375) == pytest.approx(np.sqrt(101.0 * 102.0))
