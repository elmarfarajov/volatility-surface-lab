"""Reference values the numerical engines are held to, with where each comes from.

**American puts, Longstaff and Schwartz (2001), Table 1.** Strike 40, rate 6%, no
dividends. Their finite-difference column is widely quoted as the American price.
Measured here, it is not quite that. In 15 of the 20 cases - every one-year case
and every two-year case at 20% volatility - it agrees to within 0.0005 with a
**Bermudan** put exercisable fifty times a year, the exercise schedule of their own
Monte Carlo, and lies 0.003 to 0.009 below the American put. In the other five
(two years, 40% volatility) it lies between the two.

**American puts, high precision.** The American prices are QuantLib's
``QdFpAmericanEngine`` in its high-precision scheme, which implements Andersen,
Lake and Offengenden's (2016) fixed-point method for the exercise boundary. Every
lattice and PDE engine here converges to them, and the Bermudan values are the
Crank-Nicolson engine of this package on a 1,600 x 1,600 grid. The values are
stored, not recomputed, so the tests do not depend on QuantLib.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmericanPut:
    spot: float
    vol: float
    maturity: float
    longstaff_schwartz: float  # their finite-difference column, as printed (three decimals)
    american: float  # Andersen-Lake-Offengenden, QuantLib high-precision scheme
    bermudan_50: float  # exercisable 50 times a year, Crank-Nicolson 1600 x 1600

    strike: float = 40.0
    rate: float = 0.06


#: Longstaff and Schwartz (2001), Table 1, with the American and Bermudan values computed here
LONGSTAFF_SCHWARTZ_TABLE: tuple[AmericanPut, ...] = (
    AmericanPut(36.0, 0.2, 1.0, 4.478, 4.486674, 4.477792),
    AmericanPut(36.0, 0.2, 2.0, 4.840, 4.848304, 4.840198),
    AmericanPut(36.0, 0.4, 1.0, 7.101, 7.108980, 7.101224),
    AmericanPut(36.0, 0.4, 2.0, 8.508, 8.514185, 8.506727),
    AmericanPut(38.0, 0.2, 1.0, 3.250, 3.257197, 3.250102),
    AmericanPut(38.0, 0.2, 2.0, 3.745, 3.751381, 3.744730),
    AmericanPut(38.0, 0.4, 1.0, 6.148, 6.154590, 6.147541),
    AmericanPut(38.0, 0.4, 2.0, 7.670, 7.674906, 7.667969),
    AmericanPut(40.0, 0.2, 1.0, 2.314, 2.319574, 2.314046),
    AmericanPut(40.0, 0.2, 2.0, 2.885, 2.889951, 2.884530),
    AmericanPut(40.0, 0.4, 1.0, 5.312, 5.318294, 5.311920),
    AmericanPut(40.0, 0.4, 2.0, 6.920, 6.923458, 6.917011),
    AmericanPut(42.0, 0.2, 1.0, 1.617, 1.621155, 1.616955),
    AmericanPut(42.0, 0.2, 2.0, 2.212, 2.216724, 2.212336),
    AmericanPut(42.0, 0.4, 1.0, 4.582, 4.588160, 4.582423),
    AmericanPut(42.0, 0.4, 2.0, 6.248, 6.250236, 6.244251),
    AmericanPut(44.0, 0.2, 1.0, 1.110, 1.112962, 1.109850),
    AmericanPut(44.0, 0.2, 2.0, 1.690, 1.693330, 1.689803),
    AmericanPut(44.0, 0.4, 1.0, 3.948, 3.952785, 3.947642),
    AmericanPut(44.0, 0.4, 2.0, 5.647, 5.646731, 5.641177),
)
