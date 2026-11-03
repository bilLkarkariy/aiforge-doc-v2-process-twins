"""Detecteur local APPRIS sur le split TRAINING d'AIForge-Doc v2 (recherche).

Idee : un patch repeint par GPT-Image puis recolle n'a ni le grain du scan,
ni la texture du papier, ni l'empreinte JPEG de la capture. Un petit reseau
entierement convolutif apprend, pixel par pixel, « ce morceau a ete recolle ».
Le score de page est le pic de sa carte ; la zone vient avec.

Discipline :

* **TRAINING seulement pour apprendre et regler.** TRAINING est coupe par
  `new_id` en 80 % d'apprentissage et 20 % de RESERVE ; le seuil de page est
  choisi sur la reserve ; le TEST ne sert qu'une fois, reglage gele ;
* **negatifs durs** : pour chaque faux, l'original au MEME endroit, pour que le
  reseau apprenne la trace du collage et pas le contenu (chiffres, logos) ;
* **memes trajets des deux cotes** : natif_png, natif_j85, demi_png ;
* **licence CC BY-NC-SA 4.0** : poids et seuils restent de la recherche, hors
  du depot et hors du produit. Aucune image, aucun poids n'est versionne.

    PYTHONPATH=src python entrainement/aiforge_v2_detecteur_appris.py vignettes
    PYTHONPATH=src python entrainement/aiforge_v2_detecteur_appris.py entrainer
    PYTHONPATH=src python entrainement/aiforge_v2_detecteur_appris.py reserve
    PYTHONPATH=src python entrainement/aiforge_v2_detecteur_appris.py predire-test
    PYTHONPATH=src python entrainement/aiforge_v2_detecteur_appris.py mesurer-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiforge_v2_developpement as dev
import banc_aiforge_v2 as banc
import mesurer_f06

COTE = 96
GRAINE = 20260927
PART_RESERVE = 0.2


def en_reserve(new_id: str) -> bool:
    """Coupe deterministe de TRAINING, independante de tout resultat."""
    h = int(hashlib.sha256(f"{GRAINE}-{new_id}".encode()).hexdigest()[:8], 16)
    return (h % 1000) < PART_RESERVE * 1000


def dossier_dev(ch: banc.Chemins) -> Path:
    d = ch.racine / "dev" / "appris"
    d.mkdir(parents=True, exist_ok=True)
    return d


def decoder(octets: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(octets)) as image:
        return np.asarray(image.convert("RGB")).copy()


def trajet_image(image: np.ndarray, trajet: str) -> np.ndarray:
    octets, _ = banc.appliquer_trajet(image, trajet)
    return decoder(octets)


def trajet_masque(masque: np.ndarray, trajet: str) -> np.ndarray:
    if trajet != "demi_png":
        return masque
    h, w = masque.shape
    petit = Image.fromarray(masque.astype(np.uint8) * 255).resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BOX)
    return np.asarray(petit) > 127


# --------------------------------------------------------------------------
# Vignettes
# --------------------------------------------------------------------------


def _decoupe(image: np.ndarray, masque: np.ndarray, cy: int, cx: int) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = masque.shape
    if h < COTE or w < COTE:
        return None
    y0 = int(np.clip(cy - COTE // 2, 0, h - COTE))
    x0 = int(np.clip(cx - COTE // 2, 0, w - COTE))
    return image[y0:y0 + COTE, x0:x0 + COTE], masque[y0:y0 + COTE, x0:x0 + COTE]


PETIT_PATCH_PX = 2000


def _tache_vignettes(args: tuple) -> list[tuple[np.ndarray, np.ndarray, int]]:
    falsifie_chemin, original_chemin, masque_chemin, exif, new_id = args[:5]
    # v1/v2 : 2 decoupes sur le patch, 1 au hasard. v3 : plus, et le double
    # pour les petits patchs, ou se concentraient les rates de la reserve.
    n_masque, n_hasard = (args[5], args[6]) if len(args) > 5 else (2, 1)
    rng = np.random.default_rng(int(hashlib.sha256(new_id.encode()).hexdigest()[:8], 16))
    falsifie = banc.charger_rgb(Path(falsifie_chemin))
    original = banc.charger_rgb(Path(original_chemin), orientation_exif=exif)
    masque = banc.charger_masque(Path(masque_chemin))
    sortie = []
    for code, trajet in enumerate(banc.TRAJETS):
        f = trajet_image(falsifie, trajet)
        o = trajet_image(original, trajet)
        m = trajet_masque(masque, trajet)
        if f.shape != o.shape or f.shape[:2] != m.shape:
            continue
        ys, xs = np.nonzero(m)
        h, w = m.shape
        centres = []
        n = n_masque * (2 if masque.sum() < PETIT_PATCH_PX else 1)
        if len(ys):
            for _ in range(n):
                i = int(rng.integers(len(ys)))
                centres.append((int(ys[i] + rng.integers(-24, 25)), int(xs[i] + rng.integers(-24, 25))))
        hasards = [(int(rng.integers(h)), int(rng.integers(w))) for _ in range(n_hasard)]
        vides = np.zeros_like(m)
        for cy, cx in centres:
            for img, msk in ((f, m), (o, vides)):
                d = _decoupe(img, msk, cy, cx)
                if d is not None:
                    sortie.append((d[0], d[1], code))
        for aleatoire in hasards:
            for img, msk in ((f, m), (o, vides)):
                d = _decoupe(img, msk, *aleatoire)
                if d is not None:
                    sortie.append((d[0], d[1], code))
    return sortie


def vignettes(
    ch: banc.Chemins, processus: int, fichier: str = "vignettes.npz", n_masque: int = 2, n_hasard: int = 1,
) -> dict[str, int]:
    meta = {m["new_id"]: m for m in dev.elements(ch, "training")}
    paires = [r for r in dev.lire_journal(ch, "training").values() if r["apparie"]]
    comptes = {}
    for partie in ("apprentissage", "reserve"):
        choix = [r for r in paires if en_reserve(r["new_id"]) == (partie == "reserve")]
        (dossier_dev(ch) / f"{partie}.json").write_text(
            json.dumps(sorted(r["new_id"] for r in choix)), encoding="utf-8")
        if partie == "reserve":
            comptes[partie] = len(choix)
            continue
        taches = [
            (str(banc._chemin_jeu(ch, meta[r["new_id"]]["image"])), r["original"],
             str(banc._chemin_jeu(ch, meta[r["new_id"]]["mask"])), bool(r["exif"]), r["new_id"],
             n_masque, n_hasard)
            for r in sorted(choix, key=lambda r: r["new_id"])
        ]
        images, masques, trajets = [], [], []
        with ProcessPoolExecutor(max_workers=processus) as pool:
            for lot in pool.map(_tache_vignettes, taches, chunksize=4):
                for img, msk, code in lot:
                    images.append(img)
                    masques.append(msk)
                    trajets.append(code)
        np.savez(dossier_dev(ch) / fichier, images=np.stack(images), masques=np.stack(masques),
                 trajets=np.array(trajets, dtype=np.uint8))
        comptes[partie] = len(choix)
        comptes["vignettes"] = len(images)
    return comptes


# --------------------------------------------------------------------------
# Reseau
# --------------------------------------------------------------------------


def construire_reseau():
    import torch
    from torch import nn

    def bloc(entree: int, sortie: int, pas: int = 1, dilatation: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(entree, sortie, 3, stride=pas, padding=dilatation, dilation=dilatation, bias=False),
            nn.BatchNorm2d(sortie),
            nn.ReLU(inplace=True),
        )

    class Reseau(nn.Module):
        """Entree RGB, sortie : logit de patch a 1/4 de resolution."""

        def __init__(self) -> None:
            super().__init__()
            self.corps = nn.Sequential(
                bloc(6, 32), bloc(32, 32),
                bloc(32, 64, pas=2), bloc(64, 64),
                bloc(64, 96, pas=2), bloc(96, 96, dilatation=2), bloc(96, 96, dilatation=4), bloc(96, 96, dilatation=8),
                nn.Conv2d(96, 1, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Canaux de residu : l'image moins sa version lissee, la ou vit le grain.
            lisse = nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
            return self.corps(torch.cat([x, (x - lisse) * 8.0], dim=1))

    return Reseau()


def _tenseur(images: np.ndarray):
    import torch

    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)


def entrainer(
    ch: banc.Chemins, iterations: int, lot: int, fils: int, modele: str = "reseau.pt",
    poids_trajets: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fichier: str = "vignettes.npz", depart: str | None = None, lr: float = 2e-3, poids_vignette: float = 0.0,
    peripherique: str = "cpu",
) -> dict[str, Any]:
    """`peripherique` : `cpu` (la recette publiee) ou `mps` (GPU d'un Mac). Les
    poids changent un peu d'un peripherique a l'autre ; ils sont toujours
    enregistres sur CPU, et la notation reste sur CPU."""
    import torch
    from torch import nn

    torch.manual_seed(GRAINE)
    torch.set_num_threads(fils)
    donnees = np.load(dossier_dev(ch) / fichier)
    images, masques = donnees["images"], donnees["masques"]
    n = len(images)
    # Tirage pondere par trajet : les trajets difficiles peuvent etre surrepresentes.
    probas = np.asarray(poids_trajets, dtype=np.float64)[donnees["trajets"]]
    probas /= probas.sum()
    rng = np.random.default_rng(GRAINE)
    reseau = construire_reseau()
    if depart:
        reseau.load_state_dict(torch.load(dossier_dev(ch) / depart, map_location="cpu"))
    reseau = reseau.to(peripherique)
    optimiseur = torch.optim.AdamW(reseau.parameters(), lr=lr, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.OneCycleLR(optimiseur, max_lr=lr, total_steps=iterations)
    cible_pool = nn.AvgPool2d(4)
    pos = float(masques.mean())
    poids_pos = torch.tensor(min(50.0, (1 - pos) / max(pos, 1e-6)), device=peripherique)
    debut = time.time()
    historique = []
    reseau.train()
    for it in range(iterations):
        idx = rng.choice(n, size=lot, p=probas)
        x = _tenseur(images[idx]).to(peripherique)
        # Symetries qui ne changent pas la trace : retournements.
        if rng.random() < 0.5:
            x = x.flip(3)
            m = torch.from_numpy(masques[idx][:, None, :, ::-1].copy()).float().to(peripherique)
        else:
            m = torch.from_numpy(masques[idx][:, None]).float().to(peripherique)
        cible = (cible_pool(m) > 0.5).float()
        sortie = reseau(x)
        perte = nn.functional.binary_cross_entropy_with_logits(sortie, cible, pos_weight=poids_pos)
        if poids_vignette:
            # La decision de page est un pic : on apprend aussi « y a-t-il un
            # patch dans cette vignette ? » sur le pic de la carte, ce qui donne
            # a un petit patch le meme poids qu'un grand.
            pic = sortie.amax(dim=(2, 3)).squeeze(1)
            present = (m.sum(dim=(1, 2, 3)) >= 40).float()
            perte = perte + poids_vignette * nn.functional.binary_cross_entropy_with_logits(pic, present)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        planning.step()
        if it % 100 == 0 or it == iterations - 1:
            historique.append((it, round(float(perte), 4), round(time.time() - debut)))
            print(f"iteration {it} perte {float(perte):.4f} {time.time() - debut:.0f} s", flush=True)
    torch.save({k: v.cpu() for k, v in reseau.state_dict().items()}, dossier_dev(ch) / modele)
    return {"vignettes": n, "iterations": iterations, "lot": lot, "poids_trajets": poids_trajets,
            "modele": modele, "fichier": fichier, "depart": depart, "lr": lr, "poids_vignette": poids_vignette,
            "peripherique": peripherique,
            "historique": historique[-5:]}


_RESEAU = None
# Notation sur CPU par defaut (la recette publiee). `APPRIS_PERIPHERIQUE=mps`
# la passe sur le GPU d'un Mac, un seul processus : memes poids, ecarts de
# l'ordre de 1e-5 sur les logits.
_PERIPHERIQUE = os.environ.get("APPRIS_PERIPHERIQUE", "cpu")


def _reseau_charge(chemin: str):
    global _RESEAU
    if _RESEAU is None:
        import torch

        torch.set_num_threads(1)
        reseau = construire_reseau()
        reseau.load_state_dict(torch.load(chemin, map_location="cpu"))
        reseau.to(_PERIPHERIQUE).eval()
        _RESEAU = reseau
    return _RESEAU


TUILE = 512
MARGE = 96  # > champ recepteur utile du reseau ; multiple de 4


def carte_image(image: np.ndarray, chemin_reseau: str) -> np.ndarray:
    """Carte de patch a 1/4 de resolution, calculee par tuiles avec marge.

    Les tuiles bornent la memoire (une page XFUND fait ~2000x3000 px) ; la marge
    rend le resultat identique a un passage d'un bloc, loin des bords de tuile.
    """
    import torch

    reseau = _reseau_charge(chemin_reseau)
    h, w = image.shape[:2]
    h4, w4 = (h + 3) // 4, (w + 3) // 4
    logits = np.zeros((h4, w4), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, h, TUILE):
            for x in range(0, w, TUILE):
                y0, x0 = max(0, y - MARGE), max(0, x - MARGE)
                y1, x1 = min(h, y + TUILE + MARGE), min(w, x + TUILE + MARGE)
                sortie = reseau(_tenseur(image[None, y0:y1, x0:x1]).to(_PERIPHERIQUE))[0, 0].cpu().numpy()
                oy, ox = (y - y0) // 4, (x - x0) // 4
                ty, tx = (min(h, y + TUILE) - y + 3) // 4, (min(w, x + TUILE) - x + 3) // 4
                bloc = sortie[oy:oy + ty, ox:ox + tx]
                logits[y // 4:y // 4 + bloc.shape[0], x // 4:x // 4 + bloc.shape[1]] = bloc
        lisse = torch.nn.functional.avg_pool2d(torch.from_numpy(logits)[None, None], 3, stride=1, padding=1)
    return torch.sigmoid(lisse)[0, 0].numpy()


def _logits_bruts(image: np.ndarray, chemin_reseau: str) -> np.ndarray:
    """Logits du reseau a 1/4 de resolution, par tuiles avec marge, sans lissage."""
    import torch

    reseau = _reseau_charge(chemin_reseau)
    h, w = image.shape[:2]
    logits = np.zeros(((h + 3) // 4, (w + 3) // 4), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, h, TUILE):
            for x in range(0, w, TUILE):
                y0, x0 = max(0, y - MARGE), max(0, x - MARGE)
                y1, x1 = min(h, y + TUILE + MARGE), min(w, x + TUILE + MARGE)
                sortie = reseau(_tenseur(np.ascontiguousarray(image[None, y0:y1, x0:x1])).to(_PERIPHERIQUE))[0, 0].cpu().numpy()
                oy, ox = (y - y0) // 4, (x - x0) // 4
                ty, tx = (min(h, y + TUILE) - y + 3) // 4, (min(w, x + TUILE) - x + 3) // 4
                bloc = sortie[oy:oy + ty, ox:ox + tx]
                logits[y // 4:y // 4 + bloc.shape[0], x // 4:x // 4 + bloc.shape[1]] = bloc
    return logits


def _lisser(logits: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.nn.functional.avg_pool2d(torch.from_numpy(logits)[None, None], 3, stride=1, padding=1)[0, 0].numpy()


def _logits_retournes(image: np.ndarray, chemin_reseau: str, direct: np.ndarray) -> np.ndarray:
    """Moyenne avec l'image retournee horizontalement (meme trace, autre bruit
    de prediction). Quand la largeur n'est pas multiple de 4, la carte
    retournee est decalee de moins d'une cellule ; le lissage 3x3 l'absorbe."""
    miroir = _logits_bruts(np.ascontiguousarray(image[:, ::-1]), chemin_reseau)[:, ::-1]
    return (direct + miroir[:, :direct.shape[1]]) / 2


def carte_logits(image: np.ndarray, chemin_reseau: str, retourner: bool = False) -> np.ndarray:
    """Comme `carte_image`, en logits lisses ; `retourner` moyenne avec l'image retournee."""
    direct = _logits_bruts(image, chemin_reseau)
    return _lisser(_logits_retournes(image, chemin_reseau, direct) if retourner else direct)


def stats_page(logits: np.ndarray) -> dict[str, Any]:
    """Scores de page candidats (choisis sur la reserve) et zone du pic."""
    i = int(np.argmax(logits))
    y, x = divmod(i, logits.shape[1])
    pic = float(logits[y, x])
    mediane = float(np.median(logits))
    mad = float(np.median(np.abs(logits - mediane))) * 1.4826 + 1e-3
    haut = np.sort(logits, axis=None)[-16:]
    fort = logits >= mediane + (pic - mediane) * 0.5
    ys, xs = np.nonzero(fort)
    proches = (np.abs(ys - y) < 40) & (np.abs(xs - x) < 80)
    ys, xs = ys[proches], xs[proches]
    return {
        "pic": pic,
        "contraste": pic - mediane,
        "z": (pic - mediane) / mad,
        "haut16": float(haut.mean()),
        "boite": [int(xs.min()) * 4, int(ys.min()) * 4, (int(xs.max()) + 1) * 4, (int(ys.max()) + 1) * 4],
    }


def score_page(carte: np.ndarray) -> tuple[float, list[int]]:
    """Pic de la carte lissee et sa boite en pixels (cellules de 4 px)."""
    i = int(np.argmax(carte))
    y, x = divmod(i, carte.shape[1])
    fort = carte >= float(carte[y, x]) * 0.5
    ys, xs = np.nonzero(fort)
    proches = (np.abs(ys - y) < 40) & (np.abs(xs - x) < 80)
    ys, xs = ys[proches], xs[proches]
    boite = [int(xs.min()) * 4, int(ys.min()) * 4, (int(xs.max()) + 1) * 4, (int(ys.max()) + 1) * 4]
    return float(carte[y, x]), boite


def _tache_reserve(args: tuple) -> dict[str, Any]:
    falsifie_chemin, original_chemin, masque_chemin, exif, new_id, chemin_reseau = args[:6]
    if len(args) > 6 and args[6]:
        return _tache_reserve_stats(args)
    falsifie = banc.charger_rgb(Path(falsifie_chemin))
    original = banc.charger_rgb(Path(original_chemin), orientation_exif=exif)
    masque = banc.charger_masque(Path(masque_chemin))
    sortie: dict[str, Any] = {"new_id": new_id}
    for trajet in banc.TRAJETS:
        m = trajet_masque(masque, trajet)
        for cote, image in (("ia", falsifie), ("honnete", original)):
            score, boite = score_page(carte_image(trajet_image(image, trajet), chemin_reseau))
            x0, y0, x1, y1 = boite
            sortie[f"{cote}/{trajet}"] = {
                "score": score,
                "zone_juste": bool(m[y0:y1, x0:x1].any()) if cote == "ia" else None,
            }
    return sortie


def _tache_reserve_stats(args: tuple) -> dict[str, Any]:
    falsifie_chemin, original_chemin, masque_chemin, exif, new_id, chemin_reseau = args[:6]
    falsifie = banc.charger_rgb(Path(falsifie_chemin))
    original = banc.charger_rgb(Path(original_chemin), orientation_exif=exif)
    masque = banc.charger_masque(Path(masque_chemin))
    sortie: dict[str, Any] = {"new_id": new_id}
    for trajet in banc.TRAJETS:
        m = trajet_masque(masque, trajet)
        for cote, image in (("ia", falsifie), ("honnete", original)):
            img = trajet_image(image, trajet)
            direct = _logits_bruts(img, chemin_reseau)
            cartes = {"brut": _lisser(direct), "retourne": _lisser(_logits_retournes(img, chemin_reseau, direct))}
            bloc = {}
            for nom, carte in cartes.items():
                st = stats_page(carte)
                x0, y0, x1, y1 = st.pop("boite")
                st["zone_juste"] = bool(m[y0:y1, x0:x1].any()) if cote == "ia" else None
                bloc[nom] = st
            sortie[f"{cote}/{trajet}"] = bloc
    return sortie


def reserve(ch: banc.Chemins, processus: int, modele: str = "reseau.pt", stats: bool = False) -> dict[str, Any]:
    meta = {m["new_id"]: m for m in dev.elements(ch, "training")}
    journal = dev.lire_journal(ch, "training")
    ids = json.loads((dossier_dev(ch) / "reserve.json").read_text(encoding="utf-8"))
    chemin_reseau = str(dossier_dev(ch) / modele)
    taches = [
        (str(banc._chemin_jeu(ch, meta[i]["image"])), journal[i]["original"],
         str(banc._chemin_jeu(ch, meta[i]["mask"])), bool(journal[i]["exif"]), i, chemin_reseau, stats)
        for i in ids
    ]
    suffixe = "_stats" if stats else ""
    fichier = dossier_dev(ch) / f"reserve_scores_{Path(modele).stem}{suffixe}.jsonl"
    faits = {}
    if fichier.exists():
        for ligne in fichier.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                r = json.loads(ligne)
                faits[r["new_id"]] = r
    taches = [t for t in taches if t[4] not in faits]
    with fichier.open("a", encoding="utf-8") as sortie, ProcessPoolExecutor(max_workers=processus) as pool:
        for r in pool.map(_tache_reserve, taches, chunksize=2):
            r["source"] = journal[r["new_id"]]["source"]
            sortie.write(json.dumps(r) + "\n")
            sortie.flush()
            faits[r["new_id"]] = r
    resultats = [faits[i] for i in ids]
    return {"paires": len(resultats), "fichier": str(fichier)} if stats else resume_reserve(resultats)


def seuil_pour(scores_honnetes: list[float], taux: float) -> float:
    """Plus petit seuil qui laisse au plus `taux` des honnetes au-dessus."""
    s = sorted(scores_honnetes)
    k = int(np.floor(taux * len(s)))
    return (s[-k - 1] if k < len(s) else 0.0) + 1e-6


def resume_reserve(resultats: list[dict[str, Any]]) -> dict[str, Any]:
    sortie: dict[str, Any] = {"paires": len(resultats)}
    for trajet in banc.TRAJETS:
        hon = [r[f"honnete/{trajet}"]["score"] for r in resultats]
        bloc = {}
        for taux in (0.005, 0.01):
            t = seuil_pour(hon, taux)
            rc = sum(r[f"ia/{trajet}"]["score"] >= t for r in resultats)
            zj = sum(r[f"ia/{trajet}"]["score"] >= t and r[f"ia/{trajet}"]["zone_juste"] for r in resultats)
            fa = sum(h >= t for h in hon)
            par_source = {}
            for s in sorted({r["source"] for r in resultats}):
                sous = [r for r in resultats if r["source"] == s]
                par_source[s] = [sum(r[f"ia/{trajet}"]["score"] >= t for r in sous),
                                 sum(r[f"honnete/{trajet}"]["score"] >= t for r in sous), len(sous)]
            bloc[f"fa_{taux}"] = {"seuil": round(t, 5), "rappel": rc, "zone_juste": zj, "fausse_alerte": fa,
                                  "par_source[rappel,fa,n]": par_source}
        sortie[trajet] = bloc
    return sortie


# --------------------------------------------------------------------------
# TEST : dossiers aveugles, puis cle
# --------------------------------------------------------------------------


def _tache_test(args: tuple[str, str]) -> dict[str, Any]:
    chemin, chemin_reseau = args
    with Image.open(chemin) as image:
        tableau = np.asarray(image.convert("RGB")).copy()
    score, boite = score_page(carte_image(tableau, chemin_reseau))
    return {"doc_id": Path(chemin).name, "score": score, "boite": boite}


def _tache_test_stats(args: tuple[str, str]) -> dict[str, Any]:
    chemin, chemin_reseau = args
    with Image.open(chemin) as image:
        tableau = np.asarray(image.convert("RGB")).copy()
    return {"doc_id": Path(chemin).name, **stats_page(_lisser(_logits_bruts(tableau, chemin_reseau)))}


def predire_test(ch: banc.Chemins, processus: int, modele: str = "reseau.pt", stats: bool = False) -> dict[str, int]:
    chemin_reseau = dossier_dev(ch) / modele
    suffixe = "_stats" if stats else ""
    tache = _tache_test_stats if stats else _tache_test
    faits = {}
    for population in ("appariee", "non_appariee"):
        for trajet in banc.TRAJETS:
            dossier = ch.aveugle / population / trajet
            sortie = ch.predictions / f"appris_{Path(modele).stem}{suffixe}" / population / f"{trajet}.json"
            if not dossier.is_dir() or sortie.exists():
                continue
            chemins = sorted(str(p) for p in dossier.iterdir())
            with ProcessPoolExecutor(max_workers=processus) as pool:
                predictions = list(pool.map(tache, [(c, str(chemin_reseau)) for c in chemins], chunksize=4))
            sortie.parent.mkdir(parents=True, exist_ok=True)
            sortie.write_text(json.dumps({
                "reseau_sha256": mesurer_f06.empreinte(chemin_reseau),
                "code_sha256": mesurer_f06.empreinte(Path(__file__)),
                "cle_lue": False,
                "predictions": predictions,
            }) + "\n", encoding="utf-8")
            faits[f"{population}/{trajet}"] = len(predictions)
    return faits


def _taux(k: int, n: int) -> dict[str, Any]:
    ic = mesurer_f06.wilson(k, n)
    return {"succes": k, "total": n, "taux": round(k / n, 4) if n else None,
            "ic95": [round(ic[0], 4), round(ic[1], 4)] if ic else None}


def mesurer_test(
    ch: banc.Chemins, seuils: dict[str, Any], modele: str = "reseau.pt", stats: bool = False,
) -> dict[str, Any]:
    """`seuils[trajet]` : un nombre (score v2, champ `score`) ou {"score": champ, "seuil": nombre}."""
    suffixe = "_stats" if stats else ""
    meta = {m["new_id"]: m for m in banc.elements_test(banc.lire_metadonnees(ch.metadonnees()))}
    rapport: dict[str, Any] = {
        "date": datetime.now(UTC).date().isoformat(),
        "jeu": {"depot": banc.DEPOT_HF, "revision": banc.REVISION_HF, "split": "TestingSet"},
        "commit": banc._commit(),
        "modele": modele,
        "reseau_sha256": mesurer_f06.empreinte(dossier_dev(ch) / modele),
        "seuils_geles": seuils,
        "populations": {},
    }
    for population in ("appariee", "non_appariee"):
        par_trajet = {}
        for trajet in banc.TRAJETS:
            chemin_pred = ch.predictions / f"appris_{Path(modele).stem}{suffixe}" / population / f"{trajet}.json"
            if not chemin_pred.exists():
                continue
            p = {x["doc_id"]: x for x in json.loads(chemin_pred.read_text(encoding="utf-8"))["predictions"]}
            chemin_cle = ch.cles / population / f"{trajet}.csv"
            with chemin_cle.open(encoding="utf-8") as entree:
                cle = list(csv.DictReader(entree))
            regle = seuils[trajet]
            champ, t = (regle["score"], regle["seuil"]) if isinstance(regle, dict) else ("score", regle)
            for x in p.values():
                x["score_retenu"] = x[champ]
            groupes: dict[str, list[dict[str, str]]] = defaultdict(list)
            for ligne in cle:
                groupes[ligne["source"]].append(ligne)
            groupes["toutes"] = cle
            bloc: dict[str, Any] = {"predictions_sha256": mesurer_f06.empreinte(chemin_pred),
                                    "cle_sha256": mesurer_f06.empreinte(chemin_cle)}
            for source, lignes in sorted(groupes.items()):
                ia = [x for x in lignes if x["verite"] == "ia"]
                ho = [x for x in lignes if x["verite"] == "honnete"]
                zone = 0
                for x in ia:
                    q = p[x["fichier"]]
                    if q["score_retenu"] >= t:
                        m = trajet_masque(banc.charger_masque(banc._chemin_jeu(ch, meta[x["paire"]]["mask"])), trajet)
                        x0, y0, x1, y1 = q["boite"]
                        zone += bool(m[y0:y1, x0:x1].any())
                bloc[source] = {
                    "rappel": _taux(sum(p[x["fichier"]]["score_retenu"] >= t for x in ia), len(ia)),
                    "rappel_zone_juste": _taux(zone, len(ia)),
                    "fausse_alerte": _taux(sum(p[x["fichier"]]["score_retenu"] >= t for x in ho), len(ho)),
                }
            par_trajet[trajet] = bloc
        if par_trajet:
            rapport["populations"][population] = par_trajet
    return rapport


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("etape", choices=("vignettes", "entrainer", "reserve", "predire-test", "mesurer-test"))
    p.add_argument("--processus", type=int, default=4)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--lot", type=int, default=48)
    p.add_argument("--seuils", type=Path, default=None, help="JSON {trajet: seuil} gele sur la reserve")
    p.add_argument("--modele", default="reseau.pt", help="fichier de poids sous dev/appris")
    p.add_argument("--poids-trajets", default="1,1,1", help="natif_png,natif_j85,demi_png")
    p.add_argument("--vignettes", default="vignettes.npz", help="fichier de vignettes sous dev/appris")
    p.add_argument("--n-masque", type=int, default=2)
    p.add_argument("--n-hasard", type=int, default=1)
    p.add_argument("--depart", default=None, help="poids de depart (affinage)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--poids-vignette", type=float, default=0.0)
    p.add_argument("--stats", action="store_true", help="reserve : scores candidats, avec retournement")
    p.add_argument("--peripherique", choices=("cpu", "mps"), default="cpu", help="entrainer : cpu (recette) ou mps")
    p.add_argument("--racine", type=Path, default=None)
    p.add_argument("--sortie", type=Path, default=None)
    a = p.parse_args(argv)
    ch = banc.Chemins((a.racine or banc.racine_par_defaut()).expanduser().resolve())
    if a.etape == "vignettes":
        resultat: Any = vignettes(ch, a.processus, a.vignettes, a.n_masque, a.n_hasard)
    elif a.etape == "entrainer":
        resultat = entrainer(ch, a.iterations, a.lot, a.processus, a.modele,
                             tuple(float(x) for x in a.poids_trajets.split(",")),
                             a.vignettes, a.depart, a.lr, a.poids_vignette, a.peripherique)
    elif a.etape == "reserve":
        resultat = reserve(ch, a.processus, a.modele, a.stats)
    elif a.etape == "predire-test":
        resultat = predire_test(ch, a.processus, a.modele, a.stats)
    else:
        if a.seuils is None:
            raise SystemExit("--seuils requis : le seuil se gele sur la reserve, jamais sur TEST")
        resultat = mesurer_test(ch, json.loads(a.seuils.read_text(encoding="utf-8")), a.modele, a.stats)
    texte = json.dumps(resultat, ensure_ascii=False, indent=2)
    if a.sortie:
        a.sortie.parent.mkdir(parents=True, exist_ok=True)
        a.sortie.write_text(texte + "\n", encoding="utf-8")
    print(texte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
