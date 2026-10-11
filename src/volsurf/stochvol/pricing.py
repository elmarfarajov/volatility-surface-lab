"""Heston prices by Fourier inversion, out of the money directly, to the last digit in the wings.

**Why out of the money.** COS and Gil-Pelaez compute one side and take the other from
put-call parity. A call far out of the money is then the difference of two numbers near
the forward, and its digits cancel: against a 40-digit reference, the original engines
were 47% and 110% wrong for a ten-year call worth 7e-8 of the forward.

**The contour integral** of Carr and Madan (1999), as extended by Lee (2004): for the
undiscounted price over the forward at log-moneyness ``k``,

    O(k) = e^{-alpha k} / pi * int_0^inf Re[ e^{-iuk} phi(u - i(alpha + 1)) / (alpha^2 + alpha - u^2 + i(2 alpha + 1) u) ] du

is the **call** for ``alpha > 0`` and the **put** for ``alpha < -1`` - the residues the
contour crosses are exactly put-call parity. So each side is priced on its own, and
nothing is subtracted. ``phi(-i(alpha + 1))`` must be finite, which keeps ``alpha + 1``
inside the moment strip.

**The damping** is chosen per strike as Lord and Kahl (2007) propose: minimise the size of
the integrand at ``u = 0``, ``e^{-alpha k} phi(-i(alpha + 1)) / (alpha^2 + alpha)``, over
the admissible ``alpha``. At that ``alpha`` the integrand is smooth, of one sign near the
origin, and never larger than it has to be.

**The quadrature** is composite 16-point Gauss-Legendre: panels graded geometrically towards
the origin, where poles near the contour (of the denominator as ``alpha`` nears -1 or 0, of
``phi`` near the edge of the strip) make the integrand peak, then panels short enough to
resolve the oscillation ``e^{-iuk}``, out to where the integrand's envelope - its modulus, which
does not oscillate - has fallen below ``1e-17`` of its value at the origin. (Testing the
oscillating real part at a single point stopped the integral early whenever that point
fell near a zero: a 1e-9 error on puts until it was found.)

COS (Fang and Oosterlee, 2008) and Carr and Madan's FFT are kept for comparison and speed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .characteristic import Heston, cf, cumulants, moment_strip

Array = NDArray[np.float64]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _envelope(u: Array, alpha: float, t: float, h: Heston) -> Array:
    """``|phi(u - i(alpha + 1)) / denominator|``: the integrand's size without its oscillation."""
    z = u - 1j * (alpha + 1.0)
    denominator = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    return np.asarray(np.abs(cf(z, t, h) / denominator), dtype=np.float64)


def optimal_alpha(k: float, t: float, h: Heston, strip: tuple[float, float] | None = None) -> float:
    """Lord and Kahl's damping for log-moneyness ``k``: positive for a call (``k >= 0``), below ``-1`` for a put.

    The size of the integrand at the origin is evaluated on 64 admissible values at once and
    the smallest taken: the graded quadrature is accurate for any damping inside the strip,
    so the optimum need not be found more finely.
    """
    p_minus, p_plus = strip if strip is not None else moment_strip(t, h)
    if k >= 0:
        lo, hi = 1e-2, min(p_plus - 1.0, 60.0) - 1e-2
    else:
        lo, hi = max(p_minus - 1.0, -60.0) + 1e-2, -1.0 - 1e-2
    if hi <= lo:
        return 0.5 * (lo + hi)
    # stay off the edges: at the strip's edge phi has a pole, and Lord and Kahl's criterion - pulled there by
    # e^{-alpha k} - would put the contour a hair from it, where the integrand is too sharp to integrate
    margin = 0.1 * (hi - lo)
    alphas = np.linspace(lo + margin, hi - margin, 64)
    with np.errstate(all="ignore"):
        value = cf(-1j * (alphas + 1.0), t, h).real
        size = np.where(
            (value > 0) & np.isfinite(value), -alphas * k + np.log(np.abs(value)) - np.log(np.abs(alphas**2 + alphas)), np.inf
        )
    return float(alphas[int(np.argmin(size))])


def _nodes(top: float, width: float) -> tuple[Array, Array]:
    """Gauss-Legendre nodes and weights on [0, top]: graded towards the origin, then panels of ``width``."""
    panels = int(min(np.ceil(top / width), 40000))
    # near the origin the integrand can peak sharply - a pole of the denominator as alpha nears -1 or 0, or of
    # phi as alpha nears the edge of the moment strip - so the first panel is graded geometrically down to 1e-8
    graded = np.concatenate([[0.0], np.geomspace(1e-8 * width, width, 48)])
    edges = np.concatenate([graded, np.linspace(width, top, panels + 1)[1:]])
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    return (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel(), (half[:, None] * _GL_W[None, :]).ravel()


def otm_price(log_moneyness: ArrayLike, t: float, h: Heston, tolerance: float = 1e-17, per_strike: bool = False) -> Array:
    """Undiscounted out-of-the-money option over the forward: a put for ``k < 0``, a call for ``k >= 0``.

    By default each side shares one damping - Lord and Kahl's choice at the side's median
    strike - so one set of characteristic-function values serves every strike of the
    expiry, and only ``e^{-iuk}`` differs. ``per_strike`` chooses the damping strike by strike.
    """
    ks = np.atleast_1d(np.asarray(log_moneyness, dtype=np.float64))
    strip = moment_strip(t, h)
    _, c2 = cumulants(t, h)
    scale = 1.0 / np.sqrt(max(c2, 1e-12))  # the frequency over which phi decays
    out = np.empty_like(ks)
    groups = [np.array([i]) for i in range(ks.size)] if per_strike else [np.flatnonzero(ks < 0), np.flatnonzero(ks >= 0)]
    for group in groups:
        if group.size == 0:
            continue
        k = ks[group]
        alpha = optimal_alpha(float(np.median(k)), t, h, strip)
        # truncate where the envelope - not the oscillating real part, which can be near a zero - is negligible
        head = _envelope(np.array([0.0]), alpha, t, h)[0]
        top = 4.0 * scale
        while _envelope(np.array([top]), alpha, t, h)[0] > tolerance * head and top < 1e7:
            top *= 2.0
        nodes, weights = _nodes(top, min(np.pi / max(float(np.max(np.abs(k))), 1e-3) / 2.0, scale, top / 32.0))
        z = nodes - 1j * (alpha + 1.0)
        kernel = cf(z, t, h) / (alpha * alpha + alpha - nodes**2 + 1j * (2.0 * alpha + 1.0) * nodes)
        values = (np.exp(-1j * k[:, None] * nodes[None, :]) * kernel[None, :]).real @ weights
        out[group] = np.exp(-alpha * k) / np.pi * values
    return out


def price(
    forward: float,
    strikes: ArrayLike,
    t: float,
    h: Heston,
    discount: float = 1.0,
    is_call: ArrayLike = True,
) -> Array:
    """Discounted Heston prices. The out-of-the-money side is integrated; the other adds the intrinsic value."""
    strikes = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    call = np.broadcast_to(np.asarray(is_call, dtype=bool), strikes.shape)
    k = np.log(strikes / forward)
    otm = otm_price(k, t, h) * forward
    otm_is_call = k >= 0
    intrinsic = np.where(call, np.maximum(forward - strikes, 0.0), np.maximum(strikes - forward, 0.0))
    value = np.where(call == otm_is_call, otm, otm + intrinsic)
    return np.asarray(discount * value, dtype=np.float64)


def cos_otm_price(log_moneyness: ArrayLike, t: float, h: Heston, terms: int = 256, width: float = 12.0) -> Array:
    """Fang and Oosterlee's COS method for the put, with calls from parity - the textbook form, for comparison."""
    k = np.atleast_1d(np.asarray(log_moneyness, dtype=np.float64))
    c1, c2 = cumulants(t, h)
    a, b = c1 - width * np.sqrt(c2), c1 + width * np.sqrt(c2)
    n = np.arange(terms)
    u = n * np.pi / (b - a)
    phi = cf(u, t, h) * np.exp(-1j * u * a)
    put = np.empty_like(k)
    for i, kk in enumerate(k):
        top = min(max(kk, a), b)
        # payoff coefficients of (e^k - e^x)^+ on [a, top]
        span = u * (top - a)
        chi = (np.exp(top) * (np.cos(span) + u * np.sin(span)) - np.exp(a)) / (1.0 + u * u)
        psi = np.where(n == 0, top - a, np.sin(span) / np.where(n == 0, 1.0, u))
        v = 2.0 / (b - a) * (np.exp(kk) * psi - chi)
        v[0] *= 0.5
        put[i] = float(np.sum((phi * v).real))
    call = put + 1.0 - np.exp(k)
    return np.asarray(np.where(k >= 0, call, put), dtype=np.float64)


def carr_madan_fft(t: float, h: Heston, alpha: float = 1.5, points: int = 4096, spacing: float = 0.25) -> tuple[Array, Array]:
    """Carr and Madan's FFT: undiscounted calls over the forward on the log-strike grid the FFT imposes."""
    j = np.arange(points)
    u = j * spacing
    step = 2.0 * np.pi / (points * spacing)  # log-strike spacing
    k = -0.5 * points * step + step * j
    psi = cf(u - 1j * (alpha + 1.0), t, h) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
    simpson = (3.0 + (-1.0) ** (j + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    x = np.exp(1j * u * 0.5 * points * step) * psi * spacing * simpson
    calls = np.exp(-alpha * k) / np.pi * np.fft.fft(x).real
    return np.asarray(k, dtype=np.float64), np.asarray(calls, dtype=np.float64)
