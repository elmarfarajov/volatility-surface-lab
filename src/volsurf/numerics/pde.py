"""The Black-Scholes PDE by finite differences: Crank-Nicolson, Rannacher smoothing, early exercise.

In ``x = ln S`` the Black-Scholes equation has constant coefficients,

    dV/dtau = sigma^2/2 V_xx + (r - q - sigma^2/2) V_x - r V,

where ``tau`` is time to expiry. The theta-scheme steps it from the payoff at
``tau = 0`` to today on a uniform grid in ``x``:

* ``theta = 1`` (implicit Euler): unconditionally stable and smooth, first order in time;
* ``theta = 1/2`` (Crank-Nicolson): second order in time - but the payoff's kink at the
  strike excites a high-frequency mode that Crank-Nicolson damps only slowly, and
  the Greeks near the strike oscillate;
* **Rannacher (1984)**: start with a few implicit half-steps, which damp that mode,
  then continue with Crank-Nicolson. Second order is kept and the oscillation goes.

Early exercise is applied at each exercise time by projection, ``V = max(V, payoff)``.
For a put, Brennan and Schwartz (1977) show the projected solution can be found
exactly within the tridiagonal solve, sweeping from the exercise region outwards;
here projection after each step is used, which converges at the same rate and also
handles calls and Bermudan schedules.

The grid places the strike exactly on a node and spot is read off by cubic
interpolation, with delta and gamma from the same interpolant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from .lattice import Contract

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PDEResult:
    price: float
    delta: float
    gamma: float
    grid: Array  # spot nodes
    values: Array  # today's values on the grid
    boundary: Array | None  # the early-exercise boundary through time (puts: highest spot exercised), if any
    boundary_times: Array | None


def solve(
    contract: Contract,
    space_steps: int = 400,
    time_steps: int = 400,
    theta: float = 0.5,
    rannacher_steps: int = 4,
    exercise: str | Sequence[float] = "european",
    width: float = 6.0,
) -> PDEResult:
    """Price on a uniform log-spot grid. ``rannacher_steps`` implicit half-steps start a Crank-Nicolson run."""
    c = contract
    if c.maturity <= 0:
        raise ValueError("the PDE needs time to expiry")
    sd = c.vol * np.sqrt(c.maturity)
    centre = np.log(c.strike)
    # a grid of width +-width standard deviations around the strike, wide enough for the spot too
    half = max(width * sd, abs(np.log(c.spot) - centre) + 3.0 * sd)
    dx = 2.0 * half / space_steps
    x = centre + dx * (np.arange(space_steps + 1) - space_steps // 2)
    spots = np.exp(x)
    values = c.payoff(spots)
    dt_full = c.maturity / time_steps
    drift = c.rate - c.yield_ - 0.5 * c.vol**2
    diffusion = 0.5 * c.vol**2
    # operator L V = a V_{i-1} + b V_i + c V_{i+1}
    lower = diffusion / dx**2 - drift / (2.0 * dx)
    main = -2.0 * diffusion / dx**2 - c.rate
    upper = diffusion / dx**2 + drift / (2.0 * dx)

    if exercise == "european":
        exercise_taus: set[int] = set()
    elif exercise == "american":
        exercise_taus = set(range(1, time_steps + 1))
    elif isinstance(exercise, str):
        raise ValueError(f"unknown exercise {exercise!r}")
    else:  # Bermudan dates, measured from today; tau = maturity - t
        exercise_taus = {int(round((c.maturity - t) / dt_full)) for t in exercise if 0 < t < c.maturity}

    # the time steps: Rannacher's implicit half-steps first, then theta steps of the full size
    schedule: list[tuple[float, float, bool]] = []  # (dt, theta, ends a full step)
    for k in range(time_steps):
        if theta == 0.5 and k < rannacher_steps // 2:
            schedule += [(0.5 * dt_full, 1.0, False), (0.5 * dt_full, 1.0, True)]
        else:
            schedule.append((dt_full, theta, True))

    n = space_steps + 1
    boundary: list[float] = []
    boundary_times: list[float] = []
    tau_step = 0
    tau = 0.0
    for dt, th, completes in schedule:
        tau += dt
        # boundary values at the new time (far from the strike the option is its discounted forward value)
        if c.is_call:
            low_bc, high_bc = 0.0, spots[-1] * np.exp(-c.yield_ * tau) - c.strike * np.exp(-c.rate * tau)
        else:
            low_bc, high_bc = c.strike * np.exp(-c.rate * tau) - spots[0] * np.exp(-c.yield_ * tau), 0.0
        rhs = values.copy()
        explicit = 1.0 - th
        if explicit > 0:
            rhs[1:-1] = values[1:-1] + explicit * dt * (lower * values[:-2] + main * values[1:-1] + upper * values[2:])
        ab = np.zeros((3, n))
        ab[0, 2:] = -th * dt * upper
        ab[1, 1:-1] = 1.0 - th * dt * main
        ab[2, :-2] = -th * dt * lower
        ab[1, 0] = ab[1, -1] = 1.0
        rhs[0], rhs[-1] = low_bc, high_bc
        values = solve_banded((1, 1), ab, rhs)
        if completes:
            tau_step += 1
            if tau_step in exercise_taus:
                intrinsic = c.payoff(spots)
                values = np.maximum(values, intrinsic)
                exercised = values <= intrinsic + 1e-12
                exercised &= intrinsic > 0
                if exercised.any():
                    boundary.append(float(spots[exercised].max() if not c.is_call else spots[exercised].min()))
                    boundary_times.append(c.maturity - tau)
    spline = CubicSpline(x, values)
    s = c.spot
    v = float(spline(np.log(s)))
    dv_dx = float(spline(np.log(s), 1))
    d2v_dx2 = float(spline(np.log(s), 2))
    delta = dv_dx / s
    gamma = (d2v_dx2 - dv_dx) / s**2
    return PDEResult(
        v,
        delta,
        gamma,
        spots,
        np.asarray(values, dtype=np.float64),
        np.asarray(boundary[::-1], dtype=np.float64) if boundary else None,
        np.asarray(boundary_times[::-1], dtype=np.float64) if boundary_times else None,
    )
