"""Article AIForge-Doc v2 : la v3 apprise sur les 612 jumeaux de collage scellés le 29/10 (daa3b8b).

Passage unique, annoncé par le protocole commité avant lui. Le jumeau suit le même procédé
que le faux (aller-retour LANCZOS vers 1024 x 1024 dans la fenêtre, recollé sur le rectangle
du masque + 6 px), sans changement de valeur. Si la v3 l'accuse autant que le faux, elle voit
le procédé, pas la retouche. Comptes seulement.

    .venv/bin/python recherche/2026-11-02/aiforge/jumeaux_v3.py --sortie <fichier.json> --journal <fichier.jsonl>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

V = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(V / "entrainement"), str(V / "src")]
import aiforge_v2_detecteur_appris as det
import banc_aiforge_v2 as banc
import banc_maison as bm
import mesurer_f06

SCELLE = V / "configs/banc_maison_scelle_aiforge_2026-10-29.json"
RAPPORT_V3 = V / "entrainement/runs/mesures/aiforge_v2_appris_v3_test_2026-09-27.json"
RESEAU_RECONSTRUIT = "5a909a273a987838"  # 16 premiers caractères, passage du 29/10
SEUIL_JUMEAUX = bm.SEUIL_JUMEAUX  # 0,10 : la règle de l'amendement 3, reprise telle quelle


def wilson(k: int, n: int) -> list[float]:
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney, ex aequo pour moitié."""
    gagne = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return round(gagne / (len(pos) * len(neg)), 4)


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--sortie", type=Path, required=True)
    a.add_argument("--journal", type=Path, required=True)
    a.add_argument("--processus", type=int, default=3)
    args = a.parse_args()
    ch = banc.Chemins(banc.racine_par_defaut().expanduser().resolve())
    reseau = det.dossier_dev(ch) / "reseau_v3.pt"
    if not mesurer_f06.empreinte(reseau).startswith(RESEAU_RECONSTRUIT):
        raise SystemExit("ce ne sont pas les poids du passage du 29/10")
    seuil = json.loads(RAPPORT_V3.read_text())["seuils_geles"]["natif_png"]
    champ, t = seuil["score"], seuil["seuil"]
    scelle = json.loads(SCELLE.read_text())["test"]
    paires = {p["base"]: p for p in bm._paires_aiforge()}
    for x in scelle:
        if mesurer_f06.empreinte(Path(paires[x["base"]]["jumeau"])) != x["sha256"]["jumeau"]:
            raise SystemExit(f"empreinte changée depuis le scellé : jumeau {x['base']}")
    # Les jumeaux, notés par la v3 comme le furent les faux et les originaux (journal : reprise sans refaire).
    faits = {}
    if args.journal.exists():
        faits = {r["base"]: r for r in map(json.loads, args.journal.read_text().splitlines()) if r}
    restants = [x["base"] for x in scelle if x["base"] not in faits]
    with args.journal.open("a") as j, ProcessPoolExecutor(max_workers=args.processus) as pool:
        for base, r in zip(restants, pool.map(det._tache_test_stats, [(paires[b]["jumeau"], str(reseau)) for b in restants],
                                              chunksize=2), strict=True):
            r["base"] = base
            j.write(json.dumps(r) + "\n")
            j.flush()
            faits[base] = r
    # Les faux et les originaux : le passage du 29/10, par la clé du TEST.
    pred = {x["doc_id"]: x for x in json.loads(
        (ch.predictions / "appris_reseau_v3_stats/appariee/natif_png.json").read_text())["predictions"]}
    cle = list(csv.DictReader((ch.cles / "appariee/natif_png.csv").open(encoding="utf-8")))
    par_base: dict[str, dict] = {}
    for ligne in cle:
        par_base.setdefault(ligne["paire"], {})[ligne["verite"]] = pred[ligne["fichier"]]
    origine = {x["base"]: x["origine"] for x in scelle}
    lignes = [(b, par_base[b]["ia"], par_base[b]["honnete"], faits[b]) for b in origine]
    bilan: dict = {}
    for groupe in ("toutes", "png", "jpeg"):
        g = [li for li in lignes if groupe == "toutes" or origine[li[0]] == groupe]
        n = len(g)
        fx, orig, jum = ([li[i][champ] for li in g] for i in (1, 2, 3))
        k_j = sum(s >= t for s in jum)
        touche = 0
        for b, _, _, r in g:
            if r[champ] >= t:
                m = det.trajet_masque(banc.charger_masque(Path(paires[b]["masque"])), "natif_png")
                x0, y0, x1, y1 = r["boite"]
                touche += bool(m[y0:y1, x0:x1].any())
        bilan[groupe] = {
            "paires": n,
            "faux_detectes": sum(s >= t for s in fx), "originaux_accuses": sum(s >= t for s in orig),
            "jumeaux_accuses": k_j, "jumeaux_accuses_taux": round(k_j / n, 4), "jumeaux_accuses_ic95": wilson(k_j, n),
            "jumeaux_accuses_boite_sur_le_masque": touche,
            "auc_faux_contre_jumeaux": auc(fx, jum), "auc_jumeaux_contre_originaux": auc(jum, orig),
            "paires_faux_au_dessus_du_jumeau": sum(a > b for a, b in zip(fx, jum, strict=True)),
        }
    tout = bilan["toutes"]
    verdict = ("la v3 voit la retouche au-delà du procédé" if tout["jumeaux_accuses_taux"] <= SEUIL_JUMEAUX
               else "la v3 voit le procédé : le critère mesure le collage")
    doc = {"etape": "article AIForge-Doc v2 : la v3 sur les jumeaux de collage, passage unique annoncé",
           "seuil_gele": {"score": champ, "seuil": t}, "regle": f"jumeaux accusés ≤ {SEUIL_JUMEAUX:.0%} (amendement 3)",
           "composition": dict(Counter(origine.values())), "bilan": bilan, "verdict": verdict}
    args.sortie.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(verdict, json.dumps(tout, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
