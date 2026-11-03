"""Recompute the paper's headline figures from the frozen predictions in `frozen/`.

New in this release (standard library only, no data download). It reads the frozen TEST
predictions of the rebuilt N3 (P3), the TEST keys, the per-twin scores of the twin pass (P4)
and the frozen native-PNG threshold, then prints the counts above the threshold, the three
AUCs, the paired counts, the 2x2 table and the exact McNemar test.

    python verify_frozen.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from fractions import Fraction
from math import comb
from pathlib import Path

ICI = Path(__file__).resolve().parent
PREDICTIONS = ICI / "frozen/predictions/appris_reseau_v3_stats/appariee/natif_png.json"
CLE = ICI / "frozen/cles/appariee/natif_png.csv"
JUMEAUX = ICI / "frozen/jumeaux-v3.jsonl"
RAPPORT = ICI / "entrainement/runs/mesures/aiforge_v2_appris_v3_test_2026-09-27.json"
SCRIPT = ICI / "entrainement/aiforge_v2_detecteur_appris.py"


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney statistic, ties counted one half."""
    gagne = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return gagne / (len(pos) * len(neg))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs b and c."""
    n, k = b + c, min(b, c)
    queue = sum(Fraction(comb(n, i), 2**n) for i in range(k + 1))
    return float(min(Fraction(1), 2 * queue))


def main() -> None:
    brut = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    script = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    print(f"network of the predictions: {brut['reseau_sha256'][:16]}")
    print(f"script of the predictions:  {brut['code_sha256'][:16]} "
          f"({'same as' if brut['code_sha256'] == script else 'DIFFERENT FROM'} the released script)")
    seuil = json.loads(RAPPORT.read_text(encoding="utf-8"))["seuils_geles"]["natif_png"]
    champ, t = seuil["score"], seuil["seuil"]
    pred = {x["doc_id"]: x[champ] for x in brut["predictions"]}
    paires: dict[str, dict[str, float]] = {}
    for ligne in csv.DictReader(CLE.open(encoding="utf-8")):
        paires.setdefault(ligne["paire"], {})[ligne["verite"]] = pred[ligne["fichier"]]
    jumeaux = {r["base"]: r[champ] for r in map(json.loads, JUMEAUX.read_text().splitlines()) if r}
    bases = sorted(b for b, v in paires.items() if "ia" in v and "honnete" in v)
    assert set(bases) == set(jumeaux), "the keys and the twin scores do not cover the same pairs"
    faux = [paires[b]["ia"] for b in bases]
    orig = [paires[b]["honnete"] for b in bases]
    jum = [jumeaux[b] for b in bases]
    n = len(bases)
    print(f"\npairs: {n}; score {champ}, frozen threshold {t}")
    for nom, v in (("forgeries", faux), ("originals", orig), ("twins", jum)):
        k = sum(s >= t for s in v)
        print(f"{nom:10s} flagged: {k}/{n} ({k / n:.1%})")
    print(f"\nAUC forgeries against originals: {auc(faux, orig):.4f}")
    print(f"AUC twins against originals:     {auc(jum, orig):.4f}")
    print(f"AUC forgeries against twins:     {auc(faux, jum):.4f}")
    print(f"pairs with forgery above original: {sum(a > b for a, b in zip(faux, orig))}/{n}")
    print(f"pairs with forgery above twin:     {sum(a > b for a, b in zip(faux, jum))}/{n}")
    ff = [s >= t for s in faux]
    fj = [s >= t for s in jum]
    t11 = sum(a and b for a, b in zip(ff, fj))
    t10 = sum(a and not b for a, b in zip(ff, fj))
    t01 = sum(b and not a for a, b in zip(ff, fj))
    t00 = n - t11 - t10 - t01
    print("\n2x2 at the frozen threshold     twin flagged  twin not flagged")
    print(f"forgery flagged                 {t11:12d}  {t10:16d}")
    print(f"forgery not flagged             {t01:12d}  {t00:16d}")
    print(f"exact McNemar test ({t10} against {t01}): p = {mcnemar_exact(t10, t01):.1e}")


if __name__ == "__main__":
    main()
