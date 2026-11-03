"""Article AIForge-Doc v2 : les analyses demandées par la revue du 02/11/2026, sans nouveau passage ni nouvelle donnée.

Tout est relu sur des prédictions déjà écrites : le passage P3 du 29/10 (faux et originaux du TEST, réseau v3
reconstruit) et le journal du passage P4 du 02/11 (les 612 jumeaux de collage scellés). Comptes et statistiques
seulement : aucun pixel, aucune valeur par document dans la sortie.

Trois analyses :
  1. intervalles bootstrap par triplets (faux, original, jumeau de la même paire, tirés ensemble) pour chaque AUC
     du contrôle des jumeaux, pour la différence des AUC et pour le taux de jumeaux accusés ;
  2. le tableau 2 x 2 (faux accusé ou non, jumeau accusé ou non) au seuil gelé, et le test exact de McNemar
     sur les paires discordantes ;
  3. l'AUC faux contre originaux par origine (PNG, JPEG), les deux cases vides du tableau 6 de l'article.

Même bootstrap que `auc_localisation_v3.py` : 2 000 tirages, graine 20261102, intervalle aux centiles 2,5 et 97,5,
paires dans l'ordre de la clé. L'intervalle faux contre originaux de l'ensemble doit donc redonner [0,983 ; 0,9944].

    nice -n 10 .venv/bin/python recherche/2026-11-02/aiforge/analyses_referee_v3.py --journal <jumeaux-v3.jsonl>

Le journal est celui que `jumeaux_v3.py --journal` a écrit le 02/11 (un JSON par jumeau, sans pixel).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parents[3]
ICI = Path(__file__).resolve().parent
RAPPORT_V3 = V / "entrainement/runs/mesures/aiforge_v2_appris_v3_test_2026-09-27.json"
SCELLE = V / "configs/banc_maison_scelle_aiforge_2026-10-29.json"
AUC_JSON = ICI / "auc-localisation-v3-troisieme-passage-test.json"
JUMEAUX_JSON = ICI / "jumeaux-v3-resultat.json"
RESEAU_RECONSTRUIT = "5a909a273a987838"  # 16 premiers caractères, passage du 29/10
GRAINE = 20261102
TIRAGES = 2000


def racine_donnees() -> Path:
    """Même règle que `banc_aiforge_v2.racine_par_defaut()`, sans importer le banc."""
    env = os.environ.get("AIFORGE_ROOT")
    return Path(env).expanduser() if env else Path.home() / "lumelia-data" / "aiforge_v2"


def sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def auc(pos, neg) -> float:
    """Mann-Whitney, ex aequo pour moitié, par les rangs moyens (même valeur que `auc_localisation_v3.py`)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    tout = np.concatenate([pos, neg])
    ordre = np.argsort(tout, kind="mergesort")
    tri = tout[ordre]
    debut = np.flatnonzero(np.r_[True, tri[1:] != tri[:-1]])
    fin = np.r_[debut[1:], len(tri)]
    rang_moyen = (debut + 1 + fin) / 2.0  # rangs de debut+1 à fin, moyenne
    rangs = np.empty(len(tout))
    rangs[ordre] = np.repeat(rang_moyen, fin - debut)
    n1, n0 = len(pos), len(neg)
    return float((rangs[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k: int, n: int) -> list[float]:
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def centiles(vals: list[float], n: int = TIRAGES) -> list[float]:
    """Mêmes centiles que `ic_paires` de `auc_localisation_v3.py`."""
    s = sorted(vals)
    return [round(s[int(0.025 * n)], 4), round(s[int(0.975 * n)], 4)]


def bootstrap_triplets(F: np.ndarray, O: np.ndarray, T: np.ndarray, seuil: float) -> dict[str, list[float]]:
    """Tire les triplets (faux, original, jumeau) ensemble, avec remise ; une valeur par tirage et par grandeur."""
    g = random.Random(GRAINE)
    m = len(F)
    sortie: dict[str, list[float]] = {k: [] for k in ("auc_fo", "auc_to", "auc_ft", "diff_fo_moins_ft", "taux_jumeaux")}
    for _ in range(TIRAGES):
        idx = np.fromiter((g.randrange(m) for _ in range(m)), dtype=np.int64, count=m)
        f, o, t = F[idx], O[idx], T[idx]
        a_fo, a_ft = auc(f, o), auc(f, t)
        sortie["auc_fo"].append(a_fo)
        sortie["auc_to"].append(auc(t, o))
        sortie["auc_ft"].append(a_ft)
        sortie["diff_fo_moins_ft"].append(a_fo - a_ft)
        sortie["taux_jumeaux"].append(float((t >= seuil).mean()))
    return sortie


def mcnemar_exact(b: int, c: int) -> dict:
    """Test exact bilatéral sur les paires discordantes (loi binomiale de paramètre 1/2), en entiers exacts."""
    n, k = b + c, min(b, c)
    p = min(Fraction(1), Fraction(2 * sum(math.comb(n, i) for i in range(k + 1)), 2**n))
    return {"discordantes": n, "faux_accuse_jumeau_non": b, "faux_non_jumeau_accuse": c,
            "p_bilateral": float(f"{float(p):.3e}"), "p_inferieur_a_1e_90": bool(p < Fraction(1, 10**90))}


def groupe(trip: list[dict], seuil: float) -> dict:
    F, O, T = (np.array([x[k] for x in trip]) for k in ("faux", "orig", "jum"))
    n = len(trip)
    boot = bootstrap_triplets(F, O, T, seuil)
    k_j = int((T >= seuil).sum())
    croises = Counter((bool(f >= seuil), bool(t >= seuil)) for f, t in zip(F, T, strict=True))
    return {
        "paires": n,
        "faux_detectes": int((F >= seuil).sum()), "originaux_accuses": int((O >= seuil).sum()), "jumeaux_accuses": k_j,
        "jumeaux_accuses_taux": round(k_j / n, 4), "jumeaux_accuses_ic95_wilson": wilson(k_j, n),
        "jumeaux_accuses_ic95_bootstrap_triplets": centiles(boot["taux_jumeaux"]),
        "auc_faux_contre_originaux": round(auc(F, O), 4), "auc_faux_contre_originaux_ic95_bootstrap_triplets": centiles(boot["auc_fo"]),
        "auc_jumeaux_contre_originaux": round(auc(T, O), 4), "auc_jumeaux_contre_originaux_ic95_bootstrap_triplets": centiles(boot["auc_to"]),
        "auc_faux_contre_jumeaux": round(auc(F, T), 4), "auc_faux_contre_jumeaux_ic95_bootstrap_triplets": centiles(boot["auc_ft"]),
        "difference_auc_faux_originaux_moins_faux_jumeaux": round(auc(F, O) - auc(F, T), 4),
        "difference_ic95_bootstrap_triplets": centiles(boot["diff_fo_moins_ft"]),
        "paires_faux_au_dessus_de_l_original": int((F > O).sum()), "paires_faux_au_dessus_du_jumeau": int((F > T).sum()),
        "ex_aequo_faux_jumeau": int((F == T).sum()), "ex_aequo_jumeau_original": int((T == O).sum()),
        "tableau_2x2_faux_jumeau": {
            "faux_accuse_jumeau_accuse": croises[(True, True)], "faux_accuse_jumeau_non": croises[(True, False)],
            "faux_non_jumeau_accuse": croises[(False, True)], "faux_non_jumeau_non": croises[(False, False)]},
    }


def horodatage(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--journal", type=Path, required=True, help="journal des jumeaux écrit par jumeaux_v3.py (jsonl)")
    a.add_argument("--sortie", type=Path, default=ICI / "analyses-referee-v3.json")
    args = a.parse_args()

    racine = racine_donnees()
    chemin_pred = racine / "predictions/appris_reseau_v3_stats/appariee/natif_png.json"
    chemin_cle = racine / "cles/appariee/natif_png.csv"
    seuil_gele = json.loads(RAPPORT_V3.read_text())["seuils_geles"]["natif_png"]
    champ, seuil = seuil_gele["score"], seuil_gele["seuil"]

    brut = json.loads(chemin_pred.read_text())
    if not brut["reseau_sha256"].startswith(RESEAU_RECONSTRUIT):
        raise SystemExit("les prédictions ne sont pas celles du réseau reconstruit du 29/10")
    pred = {x["doc_id"]: x for x in brut["predictions"]}
    cle = list(csv.DictReader(chemin_cle.open(encoding="utf-8")))
    par_paire: dict[str, dict[str, float]] = {}
    for ligne in cle:
        par_paire.setdefault(ligne["paire"], {})[ligne["verite"]] = pred[ligne["fichier"]][champ]
    ordre = [b for b, v in par_paire.items() if "ia" in v and "honnete" in v]  # ordre de la clé, comme auc_localisation_v3.py

    scelle = json.loads(SCELLE.read_text())["test"]
    origine = {x["base"]: x["origine"] for x in scelle}
    corpus = {x["base"]: x["source"] for x in scelle}
    journal = {r["base"]: r for r in map(json.loads, args.journal.read_text().splitlines()) if r}
    if not (set(ordre) == set(origine) == set(journal)):
        raise SystemExit("les paires de la clé, du scellé et du journal ne sont pas les mêmes")
    if any(journal[b][champ] is None for b in ordre):
        raise SystemExit("un jumeau n'a pas de score")
    trip = [{"base": b, "faux": par_paire[b]["ia"], "orig": par_paire[b]["honnete"], "jum": journal[b][champ],
             "origine": origine[b], "corpus": corpus[b]} for b in ordre]

    bilan = {"toutes": groupe(trip, seuil)}
    for o in ("png", "jpeg"):
        bilan[o] = groupe([x for x in trip if x["origine"] == o], seuil)
    par_corpus = {}
    for c in sorted({x["corpus"] for x in trip}):
        sous = [x for x in trip if x["corpus"] == c]
        F, O, T = (np.array([x[k] for x in sous]) for k in ("faux", "orig", "jum"))
        par_corpus[c] = {"paires": len(sous), "faux_detectes": int((F >= seuil).sum()), "originaux_accuses": int((O >= seuil).sum()),
                         "jumeaux_accuses": int((T >= seuil).sum()), "auc_faux_contre_originaux": round(auc(F, O), 4),
                         "auc_jumeaux_contre_originaux": round(auc(T, O), 4), "auc_faux_contre_jumeaux": round(auc(F, T), 4)}

    t2 = bilan["toutes"]["tableau_2x2_faux_jumeau"]
    b, c = t2["faux_accuse_jumeau_non"], t2["faux_non_jumeau_accuse"]
    k_ff = t2["faux_accuse_jumeau_accuse"]
    detectes = bilan["toutes"]["faux_detectes"]
    mcn = mcnemar_exact(b, c)
    part = {"detections_dont_le_jumeau_est_accuse": k_ff, "detections": detectes,
            "part": round(k_ff / detectes, 4), "ic95_wilson": wilson(k_ff, detectes),
            "detections_dont_le_jumeau_n_est_pas_accuse": b, "part_non_reproduite": round(b / detectes, 4),
            "ic95_wilson_non_reproduite": wilson(b, detectes)}

    # Les chiffres déjà publiés doivent ressortir tels quels, sinon la sortie dit qu'elle ne les reproduit pas.
    p3 = json.loads(AUC_JSON.read_text())["mesure"]["natif_png"]
    pub = json.loads(JUMEAUX_JSON.read_text())["bilan"]
    ok = {
        "p3_faux_et_originaux_detectes": (bilan["toutes"]["faux_detectes"], bilan["toutes"]["originaux_accuses"]) == (p3["detectes_au_seuil_gele"], p3["originaux_accuses"]),
        "p3_auc_faux_contre_originaux": bilan["toutes"]["auc_faux_contre_originaux"] == p3["auc"],
        "p3_ic95_bootstrap_paires_redonne": bilan["toutes"]["auc_faux_contre_originaux_ic95_bootstrap_triplets"] == p3["auc_ic95_bootstrap_paires"],
        "p3_paires_faux_au_dessus_de_l_original": bilan["toutes"]["paires_faux_au_dessus_de_l_original"] == p3["paires_ou_le_faux_score_plus_haut"],
    }
    for g in ("toutes", "png", "jpeg"):
        ok[f"p4_{g}"] = all((
            bilan[g]["jumeaux_accuses"] == pub[g]["jumeaux_accuses"],
            bilan[g]["auc_faux_contre_jumeaux"] == pub[g]["auc_faux_contre_jumeaux"],
            bilan[g]["auc_jumeaux_contre_originaux"] == pub[g]["auc_jumeaux_contre_originaux"],
            bilan[g]["paires_faux_au_dessus_du_jumeau"] == pub[g]["paires_faux_au_dessus_du_jumeau"],
            bilan[g]["jumeaux_accuses_ic95_wilson"] == pub[g]["jumeaux_accuses_ic95"]))

    st = args.journal.stat()
    doc = {
        "etape": "article AIForge-Doc v2 : analyses demandées par la revue du 02/11/2026, lues sur des prédictions déjà faites, aucun nouveau passage",
        "entrees": {
            "predictions_p3_sha256": sha256(chemin_pred), "reseau_p3_sha256_prefixe": brut["reseau_sha256"][:16],
            "cle_sha256": sha256(chemin_cle), "liste_scellee_sha256": sha256(SCELLE),
            "journal_p4_sha256": sha256(args.journal), "journal_p4_lignes": len(journal),
            "journal_p4_fichier_cree_utc": horodatage(st.st_birthtime), "journal_p4_derniere_ecriture_utc": horodatage(st.st_mtime),
            "script_jumeaux_sha256": sha256(ICI / "jumeaux_v3.py"), "protocole_sha256": sha256(ICI / "PROTOCOLE_jumeaux_v3.md")},
        "seuil_gele": {"score": champ, "seuil": seuil, "trajet": "natif_png"},
        "methode": {
            "auc": "Mann-Whitney, ex aequo pour moitié ; faux contre originaux, jumeaux contre originaux, faux contre jumeaux",
            "bootstrap_triplets": f"{TIRAGES} tirages avec remise des triplets (faux, original, jumeau) d'une même paire, graine {GRAINE}, "
                                  "intervalle aux centiles 2,5 et 97,5 ; les trois AUC, leur différence et le taux de jumeaux accusés "
                                  "sortent des mêmes tirages ; par origine, le tirage se fait dans l'origine",
            "ordre_des_paires": "ordre de la clé du TEST, comme auc_localisation_v3.py : l'intervalle faux contre originaux de l'ensemble redonne celui du rapport",
            "mcnemar": "test exact bilatéral sur les paires discordantes, binomiale de paramètre 1/2, entiers exacts",
            "wilson": "Wilson à 95 %, ignore l'appariement"},
        "reproduit": ok,
        "composition": dict(Counter(origine.values())),
        "bilan": bilan,
        "tableau_2x2_ensemble": {
            "lignes": "faux accusé / faux non accusé", "colonnes": "jumeau accusé / jumeau non accusé",
            "cases": t2, "mcnemar_exact": mcn, "detections": part},
        "par_corpus": par_corpus,
    }
    args.sortie.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"reproduit": ok, "tableau_2x2": t2, "mcnemar": mcn}, ensure_ascii=False))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
