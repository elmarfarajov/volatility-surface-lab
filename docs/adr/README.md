# Architecture decision records

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-one-generalised-model-parameterised-by-the-yield.md) | One generalised model, parameterised by the yield of the underlying | Accepted |
| [0002](0002-greeks-have-stated-units-and-limits.md) | Every Greek has stated units, a stated time convention, and a limit | Accepted |
| [0003](0003-every-result-is-held-to-references-it-does-not-share-code-with.md) | Every result is held to references it does not share code with | Accepted |
| [0004](0004-rebuilt-code-is-strictly-typed-and-its-charts-are-generated.md) | Rebuilt code is strictly typed, and its charts are generated | Accepted |
| [0005](0005-american-references-are-computed-not-copied-from-tables.md) | American references are computed to high precision, not copied from tables | Accepted |
| [0006](0006-crank-nicolson-starts-with-rannacher-steps.md) | Crank-Nicolson always starts with Rannacher's implicit half-steps | Accepted |
| [0007](0007-monte-carlo-always-states-its-error.md) | Monte Carlo always states its error, and early exercise is given as two bounds | Accepted |
| [0008](0008-tree-parameterisations-follow-their-papers.md) | Each tree follows its paper, and the differences from QuantLib are stated | Accepted |
| [0009](0009-the-black-function-is-computed-through-the-mills-ratio.md) | The Black function is computed through the Mills ratio, never as a difference of two prices | Accepted |
| [0010](0010-implied-volatility-is-inverted-on-log-price-objectives.md) | Implied volatility is inverted on log-price objectives, and its error is measured against the conditioning | Accepted |
| [0011](0011-market-data-is-snapshotted-locally-and-cleaned-for-stated-reasons.md) | Market data is snapshotted locally, and every quote is kept or dropped for a stated reason | Accepted |
| [0012](0012-no-arbitrage-conditions-are-constraints-checked-on-the-whole-line.md) | No-arbitrage conditions are hard constraints, checked on the whole real line | Accepted |
| [0013](0013-between-expiries-prices-are-interpolated.md) | Between expiries prices are interpolated, and beyond the last the distribution is convolved | Accepted |
| [0014](0014-ssvi-and-essvi-are-kept-as-arbitrage-free-starts.md) | SSVI and eSSVI are kept as arbitrage-free starting points, not as the fitted surface | Accepted |
| [0015](0015-heston-prices-out-of-the-money-by-contour-integral.md) | Heston prices out of the money by a contour integral, held to a 40-digit reference | Accepted |
| [0016](0016-the-moment-strip-comes-from-explosion-times.md) | The moment strip comes from explosion times, and closed forms are re-derived, not recalled | Accepted |
| [0017](0017-heston-calibration-reports-its-identifiability-and-simulation-uses-qe.md) | Heston calibration reports what it can identify, and simulation uses QE | Accepted |
| [0018](0018-local-volatility-in-closed-form-from-the-price-interpolation.md) | Local volatility comes in closed form from the price interpolation, with a self-similar short end | Accepted |
| [0019](0019-sabr-is-held-to-an-exact-reference.md) | SABR's expansion is held to an exact reference, not to itself | Accepted |
| [0020](0020-smile-dynamics-are-computed-in-each-model.md) | Smile dynamics are computed inside each model, and the delta follows | Accepted |
| [0021](0021-exotic-closed-forms-are-held-to-quantlib-and-monte-carlo-to-them.md) | Exotic closed forms are held to QuantLib, and Monte Carlo is held to the closed forms | Accepted |
| [0022](0022-discrete-monitoring-is-priced-on-its-own-grid.md) | Discrete monitoring is priced on its own grid; the continuity correction is checked, not assumed | Accepted |
| [0023](0023-the-arithmetic-asian-reference-is-controlled-monte-carlo.md) | The arithmetic Asian reference is Monte Carlo with the geometric control, not an analytic engine | Accepted |
| [0024](0024-exotics-are-priced-under-every-model-of-the-surface.md) | Exotics are priced under every model of the surface, and the spread is reported | Accepted |
