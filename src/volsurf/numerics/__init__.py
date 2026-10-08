"""Numerical engines for when there is no closed form: lattices, finite differences and Monte Carlo.

* :mod:`.lattice` - CRR, Jarrow-Rudd, Tian, Leisen-Reimer and trinomial trees, European,
  Bermudan and American exercise, BBS smoothing and Richardson extrapolation;
* :mod:`.pde` - the Black-Scholes PDE by the theta-scheme, Crank-Nicolson with Rannacher
  smoothing, and early exercise by projection, with the exercise boundary;
* :mod:`.montecarlo` - pseudo-random, antithetic, control-variate and randomised
  quasi-Monte Carlo (scrambled Sobol') estimates with honest errors, and Longstaff-Schwartz
  bracketed by an out-of-sample lower bound and a dual upper bound;
* :mod:`.references` - the American and Bermudan values the engines are held to.
"""

from .lattice import LATTICES, Contract, LatticeResult, bbsr
from .lattice import price as lattice_price
from .montecarlo import EarlyExercise, Estimate, european, longstaff_schwartz
from .pde import PDEResult, solve

__all__ = [
    "LATTICES",
    "Contract",
    "EarlyExercise",
    "Estimate",
    "LatticeResult",
    "PDEResult",
    "bbsr",
    "european",
    "lattice_price",
    "longstaff_schwartz",
    "solve",
]
