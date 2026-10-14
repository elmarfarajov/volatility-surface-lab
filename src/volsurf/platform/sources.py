"""Where an analysis's quotes come from, fingerprinted so that a result can be tied to its exact input.

Two sources: a stored option-chain snapshot (``volsurf chains fetch``), identified by its
ticker and day and the SHA-256 of its files; or a synthetic chain generated from a known
Heston model, identified by the model, the seed and the SHA-256 of the generated quotes.
Either way the fingerprint goes into the run's manifest, and a rerun refuses to compare
results computed from different bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..heston import HestonParams
from ..market import chains
from ..market.chains import Chain
from ..market_data import synthetic_chain

DEMO_MODEL = HestonParams(v0=0.028, kappa=2.5, theta=0.042, sigma=0.8, rho=-0.74)
DEMO_AS_OF = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)
DEMO_MATURITIES = (9, 23, 44, 72, 107, 170, 261, 352, 534, 716)


@dataclass(frozen=True)
class Source:
    kind: str  # "snapshot" or "synthetic"
    ticker: str
    day: str
    sha256: str
    chain: Chain = field(repr=False, compare=False)
    detail: dict[str, object] = field(default_factory=dict, compare=False)

    def describe(self) -> str:
        if self.kind == "snapshot":
            return f"{self.ticker} option chain stored on {self.day} (Yahoo Finance, delayed), SHA-256 {self.sha256[:12]}"
        return f"synthetic Heston chain {self.detail.get('model')}, seed {self.detail.get('seed')}, SHA-256 {self.sha256[:12]}"

    def record(self) -> dict[str, object]:
        return {"kind": self.kind, "ticker": self.ticker, "day": self.day, "sha256": self.sha256, **self.detail}


def _digest(*blobs: bytes) -> str:
    h = hashlib.sha256()
    for blob in blobs:
        h.update(hashlib.sha256(blob).digest())
    return h.hexdigest()


def snapshot(ticker: str, day: str | None = None) -> Source:
    """A stored snapshot: the latest of ``ticker`` unless ``day`` is given."""
    name = ticker.replace("^", "")
    base = chains.data_root()
    day = day or chains.latest(name, base)
    if day is None:
        raise FileNotFoundError(f"no snapshot of {ticker} under {base}; run `volsurf chains fetch {ticker}`")
    files = [base / day / f"{name}.csv.gz", base / day / f"{name}.json"]
    digest = _digest(*(f.read_bytes() for f in files))
    return Source("snapshot", ticker, day, digest, chains.load(ticker, day), {"files": [f.name for f in files]})


def synthetic(model: HestonParams = DEMO_MODEL, seed: int = 7, maturities_days: tuple[int, ...] = DEMO_MATURITIES) -> Source:
    """A chain generated from a known Heston model with quote noise: the offline demonstration, and the tests' market."""
    raw = synthetic_chain(model, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=maturities_days, seed=seed)
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp(DEMO_AS_OF))
    chain = Chain("SYN", 5000.0, DEMO_AS_OF, quotes.assign(vendor_iv=np.nan))
    digest = _digest(raw.to_csv(index=False, float_format="%.12g").encode())
    model_text = f"v0={model.v0}, kappa={model.kappa}, theta={model.theta}, sigma={model.sigma}, rho={model.rho}"
    return Source("synthetic", "SYN", DEMO_AS_OF.date().isoformat(), digest, chain, {"model": model_text, "seed": seed})
