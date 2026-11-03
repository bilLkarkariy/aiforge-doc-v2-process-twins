"""AUC et localisation de la v3 AIForge, lues sur les prédictions du passage TEST du 27/09 (aucun nouveau passage). Comptes seulement."""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

V = Path(__file__).resolve().parents[3]  # release: was the absolute path of the private repository
sys.path[:0] = [str(V / "entrainement"), str(V / "src")]
import aiforge_v2_detecteur_appris as det
import banc_aiforge_v2 as banc
import mesurer_f06

ch = banc.Chemins(banc.racine_par_defaut().expanduser().resolve())
r = json.loads((V / "entrainement/runs/mesures/aiforge_v2_appris_v3_test_2026-09-27.json").read_text())
meta = {m["new_id"]: m for m in banc.elements_test(banc.lire_metadonnees(ch.metadonnees()))}


def auc(pos, neg):
    """Mann-Whitney, ex aequo comptés pour moitié."""
    tout = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    i = 0
    somme = 0.0
    while i < len(tout):
        j = i
        while j < len(tout) and tout[j][0] == tout[i][0]:
            j += 1
        moy = (i + 1 + j) / 2
        somme += moy * sum(e for _, e in tout[i:j])
        i = j
    n1, n0 = len(pos), len(neg)
    return (somme - n1 * (n1 + 1) / 2) / (n1 * n0)


def ic_paires(paires, n=2000, graine=20261102):
    """Bootstrap sur les paires (faux, original de la même paire), 2,5 et 97,5 centiles."""
    g = random.Random(graine)
    vals = []
    for _ in range(n):
        tirage = [paires[g.randrange(len(paires))] for _ in paires]
        vals.append(auc([a for a, _ in tirage], [b for _, b in tirage]))
    vals.sort()
    return round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)


bilan = {}
for trajet in ("natif_png", "natif_j85", "demi_png"):
    chemin_pred = ch.predictions / "appris_reseau_v3_stats" / "appariee" / f"{trajet}.json"
    chemin_cle = ch.cles / "appariee" / f"{trajet}.csv"
    attendu = r["populations"]["appariee"][trajet]
    integre = (mesurer_f06.empreinte(chemin_pred) == attendu["predictions_sha256"]
               and mesurer_f06.empreinte(chemin_cle) == attendu["cle_sha256"])
    p = {x["doc_id"]: x for x in json.loads(chemin_pred.read_text())["predictions"]}
    cle = list(csv.DictReader(chemin_cle.open(encoding="utf-8")))
    regle = r["seuils_geles"][trajet]
    champ, t = regle["score"], regle["seuil"]
    par_paire = defaultdict(dict)
    for x in cle:
        par_paire[x["paire"]][x["verite"]] = p[x["fichier"]][champ]
    paires = [(v["ia"], v["honnete"]) for v in par_paire.values() if "ia" in v and "honnete" in v]
    ia = [x for x in cle if x["verite"] == "ia"]
    ho = [x for x in cle if x["verite"] == "honnete"]
    pos = [p[x["fichier"]][champ] for x in ia]
    neg = [p[x["fichier"]][champ] for x in ho]
    entree = {"empreintes_conformes_au_rapport": integre, "faux": len(pos), "originaux": len(neg),
              "paires_completes": len(paires), "score_gele": champ,
              "detectes_au_seuil_gele": sum(s >= t for s in pos), "originaux_accuses": sum(s >= t for s in neg),
              "auc": round(auc(pos, neg), 4), "auc_ic95_bootstrap_paires": ic_paires(paires),
              "paires_ou_le_faux_score_plus_haut": sum(a > b for a, b in paires)}
    if trajet == "natif_png":
        # La « zone juste » du rapport : la boîte autour du pic touche le masque. On mesure ce qu'elle vaut.
        couverture, iou, aire_boite, aire_masque, touche = [], [], [], [], 0
        for x in ia:
            q = p[x["fichier"]]
            if q[champ] < t:
                continue
            m = det.trajet_masque(banc.charger_masque(banc._chemin_jeu(ch, meta[x["paire"]]["mask"])), trajet).astype(bool)
            x0, y0, x1, y1 = q["boite"]
            inter = int(m[y0:y1, x0:x1].sum())
            ab, am = (x1 - x0) * (y1 - y0), int(m.sum())
            touche += inter > 0
            couverture.append(inter / am if am else 0.0)
            iou.append(inter / (ab + am - inter) if ab + am - inter else 0.0)
            aire_boite.append(ab / m.size)
            aire_masque.append(am / m.size)
        med = lambda v: round(sorted(v)[len(v) // 2], 3)
        entree["localisation"] = {
            "detections": len(iou), "boite_touche_le_masque": touche,
            "iou_boite_masque_mediane": med(iou), "iou_superieure_a_0_5": sum(v > 0.5 for v in iou),
            "part_du_masque_dans_la_boite_mediane": med(couverture),
            "aire_boite_part_de_l_image_mediane": med(aire_boite), "aire_masque_part_de_l_image_mediane": med(aire_masque)}
    bilan[trajet] = entree
print(json.dumps(bilan, indent=1, ensure_ascii=False))
