"""Finite differences and Monte Carlo: convergence orders, Rannacher's cure, and early exercise bracketed."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.analytic import greeks
from volsurf.numerics.lattice import Contract
from volsurf.numerics.montecarlo import european, longstaff_schwartz
from volsurf.numerics.pde import solve
from volsurf.numerics.references import LONGSTAFF_SCHWARTZ_TABLE

CALL = Contract(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)
PUT_36 = Contract(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False)
REFERENCE_36 = LONGSTAFF_SCHWARTZ_TABLE[0]


# ---------------------------------------------------------------------- finite differences
def test_crank_nicolson_is_second_order_and_implicit_euler_first():
    reference = CALL.european()
    cn = [abs(solve(CALL, n, n, 0.5).price - reference) for n in (100, 200, 400)]
    implicit = [abs(solve(CALL, n, n, 1.0).price - reference) for n in (100, 200, 400)]
    assert cn[0] / cn[1] > 3.5 and cn[1] / cn[2] > 3.5
    assert 1.6 < implicit[1] / implicit[2] < 3.0


def test_rannacher_steps_cure_crank_nicolsons_gamma_near_the_strike():
    """Large time steps and a fine grid: plain Crank-Nicolson's gamma rings at the strike; four implicit half-steps stop it."""
    contract = Contract(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, True)

    def worst_gamma_error(rannacher: int) -> float:
        result = solve(contract, 800, 25, 0.5, rannacher)
        near = (result.grid > 90) & (result.grid < 110)
        numerical = np.gradient(np.gradient(result.values, result.grid), result.grid)[near]
        exact = greeks(result.grid[near], 100.0, 0.25, 0.05, 0.0, 0.2, True).gamma
        return float(np.max(np.abs(numerical - exact)))

    assert worst_gamma_error(0) > 0.1  # bigger than gamma itself (about 0.04)
    assert worst_gamma_error(4) < 1e-4


def test_the_pde_gives_delta_and_gamma():
    result = solve(CALL, 800, 800)
    exact = greeks(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)
    assert result.delta == pytest.approx(float(exact.delta), abs=1e-5)
    assert result.gamma == pytest.approx(float(exact.gamma), abs=1e-5)


def test_american_and_bermudan_puts_reach_their_references():
    assert solve(PUT_36, 1600, 1600, exercise="american").price == pytest.approx(REFERENCE_36.american, abs=4e-4)
    dates = [k / 50 for k in range(1, 51)]
    assert solve(PUT_36, 800, 800, exercise=dates).price == pytest.approx(REFERENCE_36.bermudan_50, abs=2e-4)


def test_the_exercise_boundary_rises_towards_the_strike_as_expiry_nears():
    result = solve(PUT_36, 800, 800, exercise="american")
    assert result.boundary is not None and result.boundary_times is not None
    assert np.all(np.diff(result.boundary) >= -0.1)  # non-decreasing in time, to the grid's resolution
    # near expiry the boundary approaches the strike only like K (1 - sigma sqrt(tau |ln tau|)) (Barles et al., 1995):
    # one step (tau = 1/800) before expiry that is about 39.3, not 40
    tau = 1.0 / 800
    asymptote = 40.0 * (1 - 0.2 * np.sqrt(tau * abs(np.log(tau))))
    assert result.boundary[-1] == pytest.approx(asymptote, abs=0.15)
    assert result.boundary[0] < 34.0


# ---------------------------------------------------------------------- Monte Carlo
@pytest.mark.parametrize("sampling", ["pseudo", "antithetic", "control", "sobol"])
def test_every_sampling_brackets_black_scholes(sampling):
    estimate = european(CALL, 2**16, sampling, seed=7)
    assert abs(estimate.price - CALL.european()) < 4 * estimate.std_error


def test_scrambled_sobol_beats_pseudo_random_by_far_at_the_same_cost():
    pseudo = european(CALL, 2**18, "pseudo", seed=1)
    sobol = european(CALL, 2**18, "sobol", seed=1)
    assert sobol.std_error < pseudo.std_error / 30


def test_longstaff_schwartz_is_bracketed_by_an_independent_lower_and_a_dual_upper_bound():
    result = longstaff_schwartz(PUT_36, [k / 50 for k in range(1, 51)], 40_000, 40_000, dual_paths=1_000, inner_paths=100, seed=3)
    truth = REFERENCE_36.bermudan_50
    assert result.lower.price - 3 * result.lower.std_error <= truth <= result.upper.price + 3 * result.upper.std_error
    assert result.gap < 0.02  # the fitted rule leaves little on the table
