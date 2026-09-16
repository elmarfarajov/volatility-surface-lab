import numpy as np
import pytest

from volsurf.calibration import calibrate_heston
from volsurf.hedging import run_black_scholes_world, run_heston_world
from volsurf.heston import HestonParams


def test_calibration_recovers_hidden_parameters(clean_snapshot, true_params):
    result = calibrate_heston(clean_snapshot, max_quotes_per_expiry=20)
    assert result.rmse_vol < 3e-4
    np.testing.assert_allclose(result.params.as_array(), true_params.as_array(), rtol=0.05)
    assert set(result.error_by_expiry()["expiry"]) == {e.expiry for e in clean_snapshot.expiries}


def test_calibration_on_noisy_quotes_is_noise_limited(noisy_snapshot, true_params):
    result = calibrate_heston(noisy_snapshot, max_quotes_per_expiry=20)
    assert result.rmse_vol < 0.004
    assert result.params.rho == pytest.approx(true_params.rho, abs=0.05)
    assert result.params.v0 == pytest.approx(true_params.v0, rel=0.1)


@pytest.mark.slow
def test_global_method_agrees_with_multistart(clean_snapshot):
    result = calibrate_heston(clean_snapshot, method="global", max_quotes_per_expiry=15, global_iterations=10, population=8)
    assert result.rmse_vol < 1e-3


def test_black_scholes_world_matches_derman_kamal():
    exp = run_black_scholes_world(n_paths=20_000, rebalances=(16, 64, 256), seed=3)
    summary = exp.summary()
    for row in summary.itertuples():
        theory = exp.notes[f"derman_kamal_std_{row.rebalances}"]
        assert row.std_pnl == pytest.approx(theory, rel=0.12)
        assert abs(row.mean_pnl) < 4 * row.std_pnl / np.sqrt(row.paths)
    stds = summary["std_pnl"].to_numpy()
    assert stds[0] / stds[1] == pytest.approx(2.0, rel=0.15)


def test_rebalances_must_nest():
    with pytest.raises(ValueError):
        run_black_scholes_world(n_paths=100, rebalances=(3, 4))


def test_heston_world_hedging_ranking():
    params = HestonParams(v0=0.035, kappa=2.2, theta=0.045, sigma=0.75, rho=-0.72)
    exp = run_heston_world(params, rebalances=(7, 21), n_paths=1_500, seed=4)
    s = exp.summary().set_index(["strategy", "rebalances"])
    unhedged = s.loc[("Unhedged", 0), "std_pnl"]
    mv = s.loc[("Heston minimum-variance delta", 21), "std_pnl"]
    plain = s.loc[("Heston delta", 21), "std_pnl"]
    assert mv < plain < unhedged
    assert s.loc[("Black-Scholes delta @ implied vol", 21), "std_pnl"] < unhedged
    assert 0.1 < exp.notes["implied_vol"] < 0.3
