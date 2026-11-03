"""Process twins for the AIForge-Doc v2 TestingSet: extract of `entrainement/banc_maison.py`
from the author's private repository, at commit daa3b8b (29 October 2026), where the 612
twins were built and sealed before the twin pass.

Only the parts that build and seal the AIForge twins are kept, copied verbatim:
`_fenetre`, `_paires_aiforge`, `_tache_jumeau_aiforge`, `sceller_aiforge`, the constants
they read, and `SEUIL_JUMEAUX`, the 10% rule that `jumeaux_v3.py` reads. The rest of the
original module (a bench of the author's own documents and forgeries) is not part of
this release. Two parts are new: this docstring with the import block, which keeps only
what these functions use, and `main`, which keeps only the `sceller-aiforge` step of the
original command line. "Amendement 3" in the comments is the internal protocol of
29 October 2026, written for another network, that set the 10% rule.

    AIFORGE_ROOT=<data> python entrainement/banc_maison.py sceller-aiforge --processus 4

`sceller_aiforge` refuses to run while `configs/banc_maison_scelle_aiforge_2026-10-29.json`
exists: that file is the list sealed on 29 October. To rebuild the twins, move it aside,
run the step, and compare the SHA-256 of each twin with the sealed list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import banc_aiforge_v2 as banc
import mesurer_f06

RACINE_DEPOT = Path(__file__).resolve().parents[1]
RACINE = Path.home() / "lumelia-data" / "banc_maison"

FENETRE_GPT = 640
MARGE_COLLAGE_PX = 6


def _fenetre(masque: np.ndarray) -> tuple[int, int, int, int]:
    """Un carré autour du champ : assez de contexte pour que le modèle imite la
    page, assez petit pour que le champ y garde sa résolution."""
    ys, xs = np.nonzero(masque)
    h, w = masque.shape
    cote = min(h, w, max(FENETRE_GPT, int((xs.max() - xs.min()) * 1.6), int((ys.max() - ys.min()) * 3)))
    cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    y0, x0 = int(np.clip(cy - cote // 2, 0, h - cote)), int(np.clip(cx - cote // 2, 0, w - cote))
    return x0, y0, x0 + cote, y0 + cote



AIFORGE = Path(os.environ.get("AIFORGE_ROOT", str(Path.home() / "lumelia-data" / "aiforge_v2")))
FICHIER_SCELLE_AIFORGE = RACINE_DEPOT / "configs" / "banc_maison_scelle_aiforge_2026-10-29.json"
TAILLE_GENERATEUR = (1024, 1024)  # rendu de GPT-Image pour une fenêtre carrée
SEUIL_JUMEAUX = 0.10  # amendement 3, point 4 : au-delà, le critère 1 mesure le collage


def _paires_aiforge() -> list[dict[str, Any]]:
    """Les paires appariées du TestingSet d'AIForge-Doc v2 : faux, original, masque."""
    import aiforge_v2_developpement as dev

    ch = banc.Chemins(AIFORGE)
    meta = {m["new_id"]: m for m in dev.elements(ch, "testing")}
    sortie = []
    for new_id, r in sorted(dev.lire_journal(ch, "testing").items()):
        if not r.get("apparie"):
            continue
        sortie.append({
            "base": new_id, "source": r["source"], "exif": bool(r["exif"]),
            "origine": "png" if Path(r["original"]).suffix.lower() == ".png" else "jpeg",
            "falsifie": str(banc._chemin_jeu(ch, meta[new_id]["image"])), "original": r["original"],
            "masque": str(banc._chemin_jeu(ch, meta[new_id]["mask"])),
            "jumeau": str(RACINE / "jumeaux_aiforge" / f"{new_id}.png"),
        })
    return sortie


def _tache_jumeau_aiforge(p: dict[str, Any]) -> dict[str, Any]:
    """Amendement 3, point 4 : approximation du collage, sans changement de valeur.
    Depuis l'original décodé, un aller-retour LANCZOS vers la taille du générateur
    dans la fenêtre de notre procédé, recollé sur le rectangle du masque + 6 px."""
    original = banc.charger_rgb(Path(p["original"]), orientation_exif=p["exif"]).copy()
    masque = banc.charger_masque(Path(p["masque"]))
    ys, xs = np.nonzero(masque)
    if original.shape[:2] != masque.shape or not len(ys):
        return {"base": p["base"], "jete": "original et masque incompatibles"}
    x0, y0, x1, y1 = _fenetre(masque)
    fenetre = Image.fromarray(np.ascontiguousarray(original[y0:y1, x0:x1]))
    aller_retour = np.asarray(fenetre.resize(TAILLE_GENERATEUR, Image.Resampling.LANCZOS)
                              .resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS))
    by0, by1 = max(y0, int(ys.min()) - MARGE_COLLAGE_PX), min(y1, int(ys.max()) + 1 + MARGE_COLLAGE_PX)
    bx0, bx1 = max(x0, int(xs.min()) - MARGE_COLLAGE_PX), min(x1, int(xs.max()) + 1 + MARGE_COLLAGE_PX)
    jumeau = original.copy()
    jumeau[by0:by1, bx0:bx1] = aller_retour[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    Path(p["jumeau"]).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(jumeau).save(p["jumeau"])
    return {"base": p["base"]}


def sceller_aiforge(processus: int) -> dict[str, Any]:
    """Liste, jumeaux et empreintes, à commiter AVANT le passage TEST (amendement 3)."""
    if FICHIER_SCELLE_AIFORGE.exists():
        raise SystemExit(f"scelle deja pose : {FICHIER_SCELLE_AIFORGE.name}")
    ps = _paires_aiforge()
    with ProcessPoolExecutor(max_workers=processus) as pool:
        faits = list(pool.map(_tache_jumeau_aiforge, ps, chunksize=4))
    gardes = {f["base"] for f in faits if "jete" not in f}
    test = [{"base": p["base"], "source": p["source"], "origine": p["origine"],
             "sha256": {nom: mesurer_f06.empreinte(Path(p[nom])) for nom in ("falsifie", "original", "masque", "jumeau")}}
            for p in ps if p["base"] in gardes]
    compte = Counter((x["origine"], x["source"]) for x in test)
    FICHIER_SCELLE_AIFORGE.write_text(json.dumps({
        "date": datetime.now(UTC).isoformat(timespec="seconds"), "commit": banc._commit(), "amendement": 3,
        "jeu": "AIForge-Doc v2, TestingSet, GPT-Image-2 ; mesure seulement (CC BY-NC-SA)",
        "jumeau": "approximation : aller-retour LANCZOS 1024x1024 dans la fenetre du procede, "
                  "recolle sur le rectangle du masque + 6 px, sans changement de valeur",
        "composition": {"|".join(k): v for k, v in sorted(compte.items())},
        "jetes": len(ps) - len(test), "test": test,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"test": len(test), "jetes": len(ps) - len(test), "composition": {"|".join(k): v for k, v in compte.items()}}



def main(argv: list[str] | None = None) -> int:
    # New in this release: the original `main` serves the whole home bench; only its
    # `sceller-aiforge` step is kept.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etape", choices=("sceller-aiforge",))
    parser.add_argument("--processus", type=int, default=4)
    a = parser.parse_args(argv)
    print(json.dumps(sceller_aiforge(a.processus), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
