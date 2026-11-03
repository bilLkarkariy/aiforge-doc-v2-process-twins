"""Extract of `entrainement/mesurer_f06.py` from the author's private repository (commit
01dca7a, unchanged since): only the two functions that the released scripts use, `wilson`
and `empreinte` (SHA-256 of a file), with the constant `Z_95`, copied verbatim. The rest of
the module, a measurement harness for other benches, is not part of this release.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

Z_95 = 1.96


def wilson(succes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """Intervalle de Wilson bilateral sur une proportion."""
    if total <= 0:
        return None
    p = succes / total
    denominateur = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominateur
    demi = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominateur
    return max(0.0, centre - demi), min(1.0, centre + demi)


def empreinte(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as entree:
        for bloc in iter(lambda: entree.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()
