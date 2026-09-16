from datetime import datetime, timezone

import pytest

from volsurf.heston import HestonParams
from volsurf.market_data import build_snapshot, synthetic_chain

TRUE_PARAMS = HestonParams(v0=0.035, kappa=2.2, theta=0.045, sigma=0.75, rho=-0.72)
AS_OF = datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def true_params() -> HestonParams:
    return TRUE_PARAMS


@pytest.fixture(scope="session")
def clean_snapshot():
    raw = synthetic_chain(TRUE_PARAMS, maturities_days=(23, 72, 170, 352, 716), strikes_per_expiry=25, iv_noise=0.0)
    return build_snapshot(raw, 5000.0, AS_OF, "SYNTH", "synthetic")


@pytest.fixture(scope="session")
def noisy_snapshot():
    raw = synthetic_chain(TRUE_PARAMS, maturities_days=(23, 72, 170, 352, 716), strikes_per_expiry=25, iv_noise=0.002, seed=3)
    return build_snapshot(raw, 5000.0, AS_OF, "SYNTH", "synthetic")
