"""The run's record: what went in, which code and libraries ran, and every number that came out.

``manifest.json`` sits next to each report. It holds the source's fingerprint (the SHA-256
of the stored snapshot's files, or of the generated synthetic quotes), the configuration,
the versions of the code and the numerical libraries, the git commit when there is one,
and :attr:`Analysis.results`. ``volsurf reproduce <report dir>`` rebuilds the source,
refuses to go on if its bytes differ, reruns the analysis with the recorded configuration,
and compares every number.

Every step is seeded, so a rerun on the same machine reproduces each number to rounding. On
another platform the floating-point libraries can differ in the last digits; the default
tolerance, a relative 1e-6, allows for that and nothing more.
"""

from __future__ import annotations

import json
import platform as host
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import __version__
from ..risk import history
from .analysis import Analysis, Config, run
from .sources import Source, snapshot, synthetic

LIBRARIES = ("numpy", "scipy", "pandas", "matplotlib", "QuantLib")


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def history_digest() -> str | None:
    """SHA-256 of the local index and VIX history, which the book's historical VaR depends on."""
    if not history.available():
        return None
    base = history.history_root()
    return _hash(*(sorted(base.glob("*.csv"))))


def _hash(*files: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for f in files:
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def record(a: Analysis) -> dict[str, object]:
    return {
        "volsurf": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": host.python_version(),
        "platform": host.platform(),
        "libraries": {name: _version(name) for name in LIBRARIES},
        "commit": _commit(),
        "source": a.source.record(),
        "history": history_digest() if a.historical is not None else None,
        "config": a.config.record(),
        "timings": {k: round(v, 3) for k, v in a.timings.items()},
        "results": a.results,
    }


def write(a: Analysis, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(record(a), indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Comparison:
    name: str
    recorded: float
    rerun: float
    agrees: bool


def rebuild(source: dict[str, object]) -> Source:
    """The recorded source again, refusing different bytes."""
    if source["kind"] == "snapshot":
        rebuilt = snapshot(str(source["ticker"]), str(source["day"]))
    else:
        rebuilt = synthetic(seed=int(source["seed"]))  # type: ignore[call-overload]
    if rebuilt.sha256 != source["sha256"]:
        raise ValueError(f"the source has changed: recorded SHA-256 {source['sha256']}, found {rebuilt.sha256}")
    return rebuilt


def reproduce(directory: Path, rtol: float = 1e-6, atol: float = 1e-10, log: object = print) -> list[Comparison]:
    """Rerun the analysis recorded in ``directory`` and compare every result."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = rebuild(manifest["source"])
    if manifest.get("history") and manifest["history"] != history_digest():
        raise ValueError("the local S&P 500 and VIX history has changed since the run; its VaR cannot be reproduced")
    config = Config(**manifest["config"])
    rerun = run(source, config, log=log)  # type: ignore[arg-type]
    out = []
    for name, value in manifest["results"].items():
        new = rerun.results.get(name, float("nan"))
        agrees = bool(np.isclose(new, value, rtol=rtol, atol=atol))
        out.append(Comparison(name, float(value), float(new), agrees))
    for name in rerun.results.keys() - manifest["results"].keys():
        out.append(Comparison(name, float("nan"), rerun.results[name], False))
    return out
