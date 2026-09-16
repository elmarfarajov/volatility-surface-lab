"""Heston (1993) stochastic-volatility model.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^v,     d<W^S, W^v>_t = rho dt

Pricing engines:

* ``heston_price_cos`` - the Fourier-cosine expansion of Fang & Oosterlee (2008).
  Vectorised over strikes, exponentially convergent, used for calibration.
* ``heston_price_integration`` - Heston's original Gil-Pelaez inversion with adaptive
  quadrature. Slow but independent; used to validate the COS engine.
* ``heston_cos_sensitivities`` - price, dV/dS and dV/dv0 from the same cosine series,
  vectorised over *paths*, which makes model-based hedging simulations tractable.

The characteristic function uses the "little Heston trap" formulation of
Albrecher, Mayer, Schoutens & Tistaert (2007), which is continuous in the complex
logarithm and therefore numerically stable for long maturities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0 or self.theta <= 0 or self.kappa <= 0 or self.sigma <= 0:
            raise ValueError("Heston requires v0 >= 0 and kappa, theta, sigma > 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly inside (-1, 1)")

    @property
    def feller_ratio(self) -> float:
        """2*kappa*theta / sigma^2; >= 1 means the variance process never touches zero."""
        return 2.0 * self.kappa * self.theta / self.sigma**2

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, values: ArrayLike) -> HestonParams:
        v0, kappa, theta, sigma, rho = (float(x) for x in values)
        return cls(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)


def cf_exponents(u: ArrayLike, maturity: float, kappa: float, theta: float, sigma: float, rho: float):
    """Return (A(u), B(u)) such that E[exp(i u ln(S_T / F_T))] = exp(A(u) + B(u) * v0).

    Splitting out the v0 dependence lets one set of exponents serve thousands of
    simulated paths that differ only in their current variance.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma**2 * (iu + u * u))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * maturity)
    one_minus_ge = 1.0 - g * e
    a = (kappa * theta / sigma**2) * ((beta - d) * maturity - 2.0 * np.log(one_minus_ge / (1.0 - g)))
    b = ((beta - d) / sigma**2) * (1.0 - e) / one_minus_ge
    return a, b


def heston_cf(u: ArrayLike, maturity: float, params: HestonParams) -> np.ndarray:
    """Characteristic function of the log forward-moneyness of S_T, ln(S_T / F_T)."""
    a, b = cf_exponents(u, maturity, params.kappa, params.theta, params.sigma, params.rho)
    return np.exp(a + b * params.v0)


def expected_integrated_variance(v0: ArrayLike, maturity: float, kappa: float, theta: float) -> np.ndarray:
    return theta * maturity + (np.asarray(v0, dtype=float) - theta) * (1.0 - np.exp(-kappa * maturity)) / kappa


def log_return_variance(v0: ArrayLike, maturity: float, kappa: float, theta: float, sigma: float, rho: float) -> np.ndarray:
    """Variance of ln(S_T/F_T), used to size the COS truncation interval.

    Because ln(phi) = A(u) + B(u) v0, the second cumulant is -(A''(0) + B''(0) v0).
    The second derivatives come from a central difference of the exact exponents,
    which is accurate for every parameter set, unlike closed-form expansions that
    degrade at high vol-of-vol.
    """
    h = 1e-3
    a, b = cf_exponents(np.array([-h, h]), maturity, kappa, theta, sigma, rho)
    a2 = (a[0] + a[1]).real / (h * h)
    b2 = (b[0] + b[1]).real / (h * h)
    return np.maximum(-(a2 + b2 * np.asarray(v0, dtype=float)), 1e-12)


@dataclass(frozen=True)
class CosSensitivities:
    """Put value and sensitivities per unit strike and discount: V = D * K * put."""

    put: np.ndarray
    dput_dx: np.ndarray
    dput_dv: np.ndarray
    n_terms: int


def _choose_n_terms(
    width: float,
    maturity: float,
    v_min: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    tail_tolerance: float,
    max_terms: int,
) -> int:
    """Smallest power of two whose highest frequency has a negligible characteristic function."""
    n = 64
    while n < max_terms:
        u_max = n * np.pi / width
        a, b = cf_exponents(np.array([u_max]), maturity, kappa, theta, sigma, rho)
        if abs(np.exp(a[0] + b[0] * v_min)) < tail_tolerance:
            break
        n *= 2
    return n


def _cos_put_series(
    log_moneyness: np.ndarray,
    v0: np.ndarray,
    maturity: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    n_terms: int | None,
    truncation: float,
    with_sensitivities: bool,
    tail_tolerance: float = 1e-14,
    max_terms: int = 16384,
    term_variance_floor: float = 0.0,
) -> CosSensitivities:
    x = np.asarray(log_moneyness, dtype=float)
    v0 = np.broadcast_to(np.asarray(v0, dtype=float), x.shape)

    c1 = -0.5 * expected_integrated_variance(v0, maturity, kappa, theta)
    c2 = log_return_variance(v0, maturity, kappa, theta, sigma, rho)
    half_width = truncation * np.sqrt(np.max(c2))
    width = 2.0 * half_width
    a = x + c1 - half_width
    b = a + width

    if n_terms is None:
        n_terms = _choose_n_terms(
            width, maturity, max(float(np.min(v0)), term_variance_floor), kappa, theta, sigma, rho, tail_tolerance, max_terms
        )
    u = np.arange(n_terms) * np.pi / width
    exp_a, exp_b = cf_exponents(u, maturity, kappa, theta, sigma, rho)

    # Re[phi(u) e^{iu(x-a)}] evaluated in real arithmetic as |.| cos(arg): one real exp
    # and one cosine replace two complex exponentials and a complex product.
    magnitude = np.exp(exp_a.real[None, :] + exp_b.real[None, :] * v0[:, None])
    phase = exp_a.imag[None, :] + exp_b.imag[None, :] * v0[:, None] + u[None, :] * (x - a)[:, None]
    cos_phase = np.cos(phase)

    upper = np.maximum(np.minimum(b, 0.0), a)[:, None]
    lower = a[:, None]
    u_row = u[None, :]
    span = u_row * (upper - lower)
    sin_span = np.sin(span)
    chi = (np.exp(upper) * (np.cos(span) + u_row * sin_span) - np.exp(lower)) / (1.0 + u_row * u_row)
    u_safe = u.copy()
    u_safe[0] = 1.0
    psi = sin_span / u_safe[None, :]
    psi[:, 0] = (upper - lower)[:, 0]
    weighted = magnitude * ((2.0 / width) * (psi - chi))
    weighted[:, 0] *= 0.5

    put = np.sum(weighted * cos_phase, axis=1)
    if not with_sensitivities:
        return CosSensitivities(put, np.full_like(put, np.nan), np.full_like(put, np.nan), n_terms)
    sin_phase = np.sin(phase)
    dput_dx = -np.sum(weighted * sin_phase * u_row, axis=1)
    dput_dv = np.sum(weighted * (exp_b.real[None, :] * cos_phase - exp_b.imag[None, :] * sin_phase), axis=1)
    return CosSensitivities(put, dput_dx, dput_dv, n_terms)


def heston_price_cos(
    forward: float,
    strikes: ArrayLike,
    maturity: float,
    discount: float,
    params: HestonParams,
    is_call: ArrayLike = True,
    n_terms: int | None = None,
    truncation: float = 20.0,
    tail_tolerance: float = 1e-14,
) -> np.ndarray:
    """European prices for many strikes of one expiry, in forward form."""
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), strikes.shape)
    x = np.log(forward / strikes)
    series = _cos_put_series(
        x,
        np.full(strikes.shape, params.v0),
        maturity,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        n_terms,
        truncation,
        with_sensitivities=False,
        tail_tolerance=tail_tolerance,
    )
    put = discount * strikes * np.maximum(series.put, 0.0)
    put = np.maximum(put, discount * np.maximum(strikes - forward, 0.0))
    call = put + discount * (forward - strikes)
    return np.where(is_call, call, put)


def heston_price_cos_spot(
    spot: float,
    strikes: ArrayLike,
    maturity: float,
    rate: float,
    dividend_yield: float,
    params: HestonParams,
    is_call: ArrayLike = True,
    **kwargs,
) -> np.ndarray:
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    return heston_price_cos(forward, strikes, maturity, np.exp(-rate * maturity), params, is_call, **kwargs)


@dataclass(frozen=True)
class PathSensitivities:
    price: np.ndarray
    delta: np.ndarray
    dprice_dv: np.ndarray


def heston_cos_sensitivities(
    spot: ArrayLike,
    variance: ArrayLike,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    is_call: bool = True,
    truncation: float = 12.0,
    tail_tolerance: float = 1e-8,
    n_buckets: int = 12,
) -> PathSensitivities:
    """Price, delta and variance-sensitivity for one contract across many (S, v) states at once.

    With the cosine interval held fixed, d/dx of exp(iu(x-a)) is iu and d/dv0 of the
    characteristic function is B(u), so all three quantities come out of one series.

    States are grouped into variance quantile buckets. Each bucket gets an interval
    sized to its own largest variance and a term count sized to its own smallest, so
    a handful of near-zero-variance paths no longer force thousands of terms on all.
    """
    spot = np.atleast_1d(np.asarray(spot, dtype=float))
    variance = np.maximum(np.broadcast_to(np.asarray(variance, dtype=float), spot.shape), 0.0)
    discount = np.exp(-rate * maturity)
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    x = np.log(forward / strike)

    put_unit = np.empty_like(spot)
    dput_dx = np.empty_like(spot)
    dput_dv = np.empty_like(spot)
    order = np.argsort(variance)
    for bucket in np.array_split(order, min(n_buckets, spot.size)):
        if bucket.size == 0:
            continue
        series = _cos_put_series(
            x[bucket],
            variance[bucket],
            maturity,
            kappa,
            theta,
            sigma,
            rho,
            None,
            truncation,
            with_sensitivities=True,
            tail_tolerance=tail_tolerance,
            max_terms=4096,
            term_variance_floor=2e-3,
        )
        put_unit[bucket] = series.put
        dput_dx[bucket] = series.dput_dx
        dput_dv[bucket] = series.dput_dv

    scale = discount * strike
    put = scale * put_unit
    put_delta = scale * dput_dx / spot
    dv = scale * dput_dv
    if is_call:
        price = put + spot * np.exp(-dividend_yield * maturity) - scale
        delta = put_delta + np.exp(-dividend_yield * maturity)
    else:
        price, delta = put, put_delta
    return PathSensitivities(price=price, delta=delta, dprice_dv=dv)


def heston_price_integration(
    forward: float,
    strike: float,
    maturity: float,
    discount: float,
    params: HestonParams,
    is_call: bool = True,
) -> float:
    """Gil-Pelaez inversion: C = D (F P1 - K P2). Independent reference engine."""
    log_k = np.log(strike / forward)

    def p1_integrand(u: float) -> float:
        return float((np.exp(-1j * u * log_k) * heston_cf(u - 1j, maturity, params) / (1j * u)).real)

    def p2_integrand(u: float) -> float:
        return float((np.exp(-1j * u * log_k) * heston_cf(u, maturity, params) / (1j * u)).real)

    opts = dict(limit=2000, epsabs=1e-12, epsrel=1e-10)
    p1 = 0.5 + integrate.quad(p1_integrand, 0.0, np.inf, **opts)[0] / np.pi
    p2 = 0.5 + integrate.quad(p2_integrand, 0.0, np.inf, **opts)[0] / np.pi
    call = discount * (forward * p1 - strike * p2)
    return call if is_call else call - discount * (forward - strike)
