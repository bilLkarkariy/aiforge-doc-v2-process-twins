"""AIForge-Doc v2: download, pixel pairing and blind TEST folders. Extract of
`entrainement/banc_aiforge_v2.py` from the author's private repository (commit 3fe265f,
unchanged since), used by the learned detector in `aiforge_v2_detecteur_appris.py`.

The original module also scored the blind TEST folders with another detector of the
author, which is not part of this release. Those steps (`predire`, `mesurer` and the
report writers `_taux`, `ventiler`, `_sha256`, `_pct`, `markdown`, `ecrire_rapport`) are
removed, with the two constants that only they used (`VERSION`, `PROTOCOLE_ID`).
Everything else is verbatim, except this docstring and the command line at the end,
which keeps the steps `telecharger`, `sources`, `apparier` and `preparer`.

Pairing is proved by the pixels: the benchmark does not ship the originals, so they are
taken from the four source corpora (CORD v2, WildReceipt, SROIE, XFUND), and a pair is
accepted only if the original equals the forgery outside the ground-truth mask.

Usage (a Hugging Face token in `HF_TOKEN`, the dataset's conditions accepted on that
account; data under `AIFORGE_ROOT` or `--racine`):

    python entrainement/banc_aiforge_v2.py tout
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

RACINE_DEPOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE_DEPOT / "src"))
sys.path.insert(0, str(RACINE_DEPOT / "entrainement"))

import mesurer_f06


DEPOT_HF = "Scam-AI/AIForge-Doc-v2"
# Revision lue le 27/09/2026 sur l'API Hugging Face. Epinglee : un jeu qui
# bouge sous le banc rendrait le chiffre non rejouable.
REVISION_HF = "1c42e42824918836255f6d9c604c9bc78ead6e08"
GENERATEUR = "openai-gpt-image-2"

TRAJETS = ("natif_png", "natif_j85", "demi_png")
QUALITE_J85 = 85
GRAINE_AVEUGLE = 20260927

# Tolerances d'appariement, fixees avant le premier appariement. Elles
# absorbent les ecarts de decodeur JPEG, pas une retouche : une retouche vit
# dans le masque, que l'on exclut avec une marge.
MARGE_MASQUE_PX = 4
ECART_MOYEN_MAX = 1.5
SEUIL_PIXEL_DIFFERENT = 12
PART_PIXELS_DIFFERENTS_MAX = 0.005

SOURCES_CORD = (
    "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
    "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    "data/train-00001-of-00004-7dbbe248962764c5.parquet",
    "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
    "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
)
URL_WILDRECEIPT = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
DEPOT_SROIE = "podbilabs/sroie-donut"
URL_XFUND = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/{langue}.{split}.zip"
LANGUES_XFUND = ("de", "es", "fr", "it", "ja", "pt", "zh")
EXTENSIONS_SOURCE = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class BancImpossible(RuntimeError):
    """Le banc refuserait sinon de produire un chiffre ambigu."""


# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------


def racine_par_defaut() -> Path:
    env = os.environ.get("AIFORGE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "lumelia-data" / "aiforge_v2"


class Chemins:
    def __init__(self, racine: Path) -> None:
        self.racine = racine
        self.jeu = racine / "jeu"
        self.sources = racine / "sources"
        self.telechargements = racine / "telechargements"
        self.apparies = racine / "apparies"
        self.aveugle = racine / "aveugle"
        self.cles = racine / "cles"
        self.predictions = racine / "predictions"
        self.journal_appariement = racine / "appariement.jsonl"

    def metadonnees(self) -> Path:
        return self.jeu / "metadata.jsonl"


# --------------------------------------------------------------------------
# Telechargement
# --------------------------------------------------------------------------


def _jeton() -> str:
    jeton = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not jeton:
        raise BancImpossible(
            "HF_TOKEN absent : ajouter le jeton dans les variables de l'environnement "
            "et accepter les conditions du jeu sur huggingface.co/datasets/" + DEPOT_HF
        )
    return jeton


def _telecharger_url(
    url: str, cible: Path, *, entetes: dict[str, str] | None = None, essais: int = 5
) -> Path:
    """Telecharge vers `cible` via un fichier `.part`, avec reprise sur erreur reseau."""
    import httpx

    if cible.exists() and cible.stat().st_size > 0:
        return cible
    cible.parent.mkdir(parents=True, exist_ok=True)
    partiel = cible.with_name(cible.name + ".part")
    attente = 2.0
    derniere: Exception | None = None
    for _ in range(essais):
        try:
            with httpx.stream(
                "GET", url, headers=entetes or {}, timeout=120, follow_redirects=True
            ) as reponse:
                if reponse.status_code in (401, 403):
                    raise BancImpossible(
                        f"acces refuse ({reponse.status_code}) sur {url} : jeton absent, "
                        "invalide, ou conditions du jeu non acceptees"
                    )
                reponse.raise_for_status()
                with partiel.open("wb") as sortie:
                    for bloc in reponse.iter_bytes(1024 * 1024):
                        sortie.write(bloc)
            partiel.replace(cible)
            return cible
        except BancImpossible:
            raise
        except (httpx.HTTPError, OSError) as exc:  # reseau : on reessaie
            derniere = exc
            time.sleep(attente)
            attente *= 2
    raise BancImpossible(f"telechargement impossible : {url} ({derniere})")


def _url_hf(depot: str, chemin: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{depot}/resolve/{revision}/{chemin}"


def lire_metadonnees(chemin: Path) -> list[dict[str, Any]]:
    lignes = []
    with chemin.open(encoding="utf-8") as entree:
        for numero, ligne in enumerate(entree, 1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                lignes.append(json.loads(ligne))
            except json.JSONDecodeError as exc:
                raise BancImpossible(f"metadata.jsonl ligne {numero} illisible : {exc}") from exc
    return lignes


def elements_test(metadonnees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test = [m for m in metadonnees if str(m.get("split", "")).lower() == "testing"]
    if not test:
        raise BancImpossible("aucune ligne `split=testing` dans metadata.jsonl")
    for m in test:
        for champ in ("new_id", "image", "mask", "source_dataset"):
            if not m.get(champ):
                raise BancImpossible(f"metadata TEST : champ {champ} absent ({m.get('new_id')})")
    return sorted(test, key=lambda m: m["new_id"])


def telecharger(ch: Chemins, *, fils: int = 8) -> dict[str, Any]:
    entetes = {"Authorization": f"Bearer {_jeton()}"}
    for fichier in ("metadata.jsonl", "dataset_info.json", "README.md"):
        _telecharger_url(_url_hf(DEPOT_HF, fichier, REVISION_HF), ch.jeu / fichier, entetes=entetes)
    test = elements_test(lire_metadonnees(ch.metadonnees()))
    chemins = [p for m in test for p in (m["image"], m["mask"])]

    def un(chemin: str) -> Path:
        relatif = chemin.split("AIForge-Doc-v2.0/")[-1].lstrip("/")
        return _telecharger_url(
            _url_hf(DEPOT_HF, relatif, REVISION_HF), ch.jeu / relatif, entetes=entetes
        )

    with ThreadPoolExecutor(max_workers=fils) as pool:
        list(pool.map(un, chemins))
    return {"elements_test": len(test), "fichiers": len(chemins)}


def _chemin_jeu(ch: Chemins, relatif: str) -> Path:
    return ch.jeu / relatif.split("AIForge-Doc-v2.0/")[-1].lstrip("/")


# --------------------------------------------------------------------------
# Bases sources : tout finit en fichiers image sous sources/<base>/
# --------------------------------------------------------------------------


def _extraire_cord(ch: Chemins) -> None:
    import pyarrow.parquet as pq

    dossier = ch.sources / "cord"
    dossier.mkdir(parents=True, exist_ok=True)
    for chemin in SOURCES_CORD:
        local = _telecharger_url(
            _url_hf("naver-clova-ix/cord-v2", chemin), ch.telechargements / "cord" / Path(chemin).name
        )
        split_fichier = Path(chemin).name.split("-")[0]
        table = pq.ParquetFile(local)
        rang = 0
        for lot in table.iter_batches(batch_size=16, columns=["image", "ground_truth"]):
            for image, verite in zip(lot.column("image").to_pylist(), lot.column("ground_truth").to_pylist()):
                meta = {}
                try:
                    meta = json.loads(verite).get("meta", {})
                except (TypeError, json.JSONDecodeError):
                    pass
                split = str(meta.get("split") or split_fichier)
                ident = meta.get("image_id", rang)
                octets = image.get("bytes") if isinstance(image, dict) else None
                if not octets:
                    rang += 1
                    continue
                extension = Path(image.get("path") or "x.png").suffix.lower() or ".png"
                if extension not in EXTENSIONS_SOURCE:
                    extension = ".png"
                cible = dossier / f"{split}_{ident}{extension}"
                if not cible.exists():
                    cible.write_bytes(octets)
                rang += 1


def _extraire_wildreceipt(ch: Chemins) -> None:
    dossier = ch.sources / "wildreceipt"
    if dossier.exists() and any(dossier.rglob("*.jpeg")):
        return
    local = _telecharger_url(URL_WILDRECEIPT, ch.telechargements / "wildreceipt.tar")
    dossier.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local) as archive:
        for membre in archive.getmembers():
            if membre.isfile() and Path(membre.name).suffix.lower() in EXTENSIONS_SOURCE:
                # Le chemin relatif garde les sous-dossiers : l'identifiant
                # WildReceipt est un chemin, pas un nom de fichier unique.
                relatif = Path(*[p for p in Path(membre.name).parts if p not in ("", "..")])
                cible = dossier / relatif
                if cible.exists():
                    continue
                cible.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(membre)
                assert source is not None
                cible.write_bytes(source.read())


def _extraire_sroie(ch: Chemins) -> None:
    import httpx

    dossier = ch.sources / "sroie"
    reponse = httpx.get(
        f"https://huggingface.co/api/datasets/{DEPOT_SROIE}", timeout=60, follow_redirects=True
    )
    reponse.raise_for_status()
    fichiers = [
        s["rfilename"] for s in reponse.json().get("siblings", [])
        if Path(s["rfilename"]).suffix.lower() in EXTENSIONS_SOURCE
    ]
    if not fichiers:
        raise BancImpossible(f"miroir SROIE {DEPOT_SROIE} : aucune image listee")

    def un(relatif: str) -> Path:
        return _telecharger_url(_url_hf(DEPOT_SROIE, relatif), dossier / relatif)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(un, fichiers))


def _extraire_xfund(ch: Chemins) -> None:
    dossier = ch.sources / "xfund"
    dossier.mkdir(parents=True, exist_ok=True)
    for langue in LANGUES_XFUND:
        for split in ("train", "val"):
            local = _telecharger_url(
                URL_XFUND.format(langue=langue, split=split),
                ch.telechargements / "xfund" / f"{langue}.{split}.zip",
            )
            with zipfile.ZipFile(local) as archive:
                for nom in archive.namelist():
                    if Path(nom).suffix.lower() not in EXTENSIONS_SOURCE:
                        continue
                    cible = dossier / Path(nom).name
                    if not cible.exists():
                        cible.write_bytes(archive.read(nom))


EXTRACTEURS: dict[str, Callable[[Chemins], None]] = {
    "cord": _extraire_cord,
    "wildreceipt": _extraire_wildreceipt,
    "sroie": _extraire_sroie,
    "xfund": _extraire_xfund,
}


def normaliser_source(nom: str) -> str:
    n = nom.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if n.startswith("cord"):
        return "cord"
    if n.startswith("wild"):
        return "wildreceipt"
    if n.startswith("sroie"):
        return "sroie"
    if n.startswith("xfund"):
        return "xfund"
    raise BancImpossible(f"base source inconnue : {nom!r}")


def sources(ch: Chemins) -> dict[str, int]:
    test = elements_test(lire_metadonnees(ch.metadonnees()))
    besoins = Counter(normaliser_source(m["source_dataset"]) for m in test)
    for base in sorted(besoins):
        EXTRACTEURS[base](ch)
    return {
        base: sum(1 for p in (ch.sources / base).rglob("*") if p.suffix.lower() in EXTENSIONS_SOURCE)
        for base in sorted(besoins)
    }


# --------------------------------------------------------------------------
# Appariement par les pixels
# --------------------------------------------------------------------------


def charger_rgb(chemin: Path, *, orientation_exif: bool = False) -> np.ndarray:
    with Image.open(chemin) as image:
        if orientation_exif:
            image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def charger_masque(chemin: Path) -> np.ndarray:
    with Image.open(chemin) as image:
        return np.asarray(image.convert("L")) > 127


def dilater(masque: np.ndarray, marge: int) -> np.ndarray:
    """Dilatation carree sans scipy : decalages successifs d'un pixel."""
    sortie = masque.copy()
    for _ in range(marge):
        agrandi = sortie.copy()
        agrandi[1:, :] |= sortie[:-1, :]
        agrandi[:-1, :] |= sortie[1:, :]
        agrandi[:, 1:] |= sortie[:, :-1]
        agrandi[:, :-1] |= sortie[:, 1:]
        sortie = agrandi
    return sortie


def comparer_hors_masque(
    original: np.ndarray, falsifie: np.ndarray, masque: np.ndarray
) -> dict[str, Any]:
    if original.shape != falsifie.shape or masque.shape != falsifie.shape[:2]:
        return {"compatible": False, "raison": "dimensions"}
    hors = ~dilater(masque, MARGE_MASQUE_PX)
    if not hors.any():
        return {"compatible": False, "raison": "masque couvrant"}
    ecart = np.abs(original.astype(np.int16) - falsifie.astype(np.int16)).max(axis=2)[hors]
    moyen = float(ecart.mean())
    part = float((ecart > SEUIL_PIXEL_DIFFERENT).mean())
    return {
        "compatible": moyen <= ECART_MOYEN_MAX and part <= PART_PIXELS_DIFFERENTS_MAX,
        "ecart_moyen": round(moyen, 4),
        "part_pixels_differents": round(part, 6),
        "raison": None,
    }


def _vignette(image: np.ndarray, cote: int = 48) -> np.ndarray:
    gris = Image.fromarray(image).convert("L").resize((cote, cote), Image.Resampling.BILINEAR)
    return np.asarray(gris, dtype=np.float32)


class IndexSource:
    """Fichiers d'une base, retrouvables par nom ou par dimensions."""

    def __init__(self, dossier: Path) -> None:
        self.fichiers = sorted(
            p for p in dossier.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS_SOURCE
        )
        self.par_cle: dict[str, list[Path]] = defaultdict(list)
        self._tailles: dict[Path, tuple[int, int]] | None = None
        self.par_taille: dict[tuple[int, int], list[Path]] = defaultdict(list)
        self._vignettes: dict[Path, np.ndarray] = {}
        for p in self.fichiers:
            for cle in _cles_nom(p.relative_to(dossier).as_posix()):
                self.par_cle[cle].append(p)

    def _indexer_tailles(self) -> None:
        if self._tailles is not None:
            return
        self._tailles = {}
        for p in self.fichiers:
            try:
                with Image.open(p) as image:
                    taille = image.size
            except OSError:
                continue
            self._tailles[p] = taille
            # Une orientation EXIF peut tourner l'image au decodage : on range
            # le fichier sous ses deux tailles possibles.
            self.par_taille[taille].append(p)
            if taille[0] != taille[1]:
                self.par_taille[(taille[1], taille[0])].append(p)

    def vignette(self, chemin: Path) -> np.ndarray:
        if chemin not in self._vignettes:
            self._vignettes[chemin] = _vignette(charger_rgb(chemin))
        return self._vignettes[chemin]

    def candidats_nom(self, image_id: Any) -> list[Path]:
        trouves: list[Path] = []
        for cle in _cles_nom(str(image_id)):
            for p in self.par_cle.get(cle, []):
                if p not in trouves:
                    trouves.append(p)
        return trouves

    def candidats_taille(self, largeur: int, hauteur: int) -> list[Path]:
        self._indexer_tailles()
        return list(dict.fromkeys(self.par_taille.get((largeur, hauteur), [])))


def _cles_nom(texte: str) -> list[str]:
    """Cles de rapprochement d'un identifiant : chemin, nom, radical."""
    t = texte.strip().replace("\\", "/").lower()
    if not t:
        return []
    nom = t.rsplit("/", 1)[-1]
    radical = nom.rsplit(".", 1)[0] if "." in nom else nom
    cles = [t, nom, radical]
    if "." in t.rsplit("/", 1)[-1]:
        cles.append(t.rsplit(".", 1)[0])
    return list(dict.fromkeys(cles))


def apparier_un(
    element: dict[str, Any], falsifie_chemin: Path, masque_chemin: Path, index: IndexSource
) -> dict[str, Any]:
    falsifie = charger_rgb(falsifie_chemin)
    masque = charger_masque(masque_chemin)
    hauteur, largeur = falsifie.shape[:2]
    essais: list[dict[str, Any]] = []

    def tester(candidats: Iterable[Path], voie: str) -> dict[str, Any] | None:
        meilleur = None
        for candidat in candidats:
            for exif in (False, True):
                try:
                    original = charger_rgb(candidat, orientation_exif=exif)
                except OSError:
                    continue
                verdict = comparer_hors_masque(original, falsifie, masque)
                essais.append({"voie": voie, "exif": exif, **{k: v for k, v in verdict.items() if k != "compatible"}})
                if verdict["compatible"] and (
                    meilleur is None or verdict["ecart_moyen"] < meilleur["ecart_moyen"]
                ):
                    meilleur = {**verdict, "original": str(candidat), "exif": exif, "voie": voie}
        return meilleur

    trouve = tester(index.candidats_nom(element.get("image_id", "")), "identifiant")
    if trouve is None:
        par_taille = index.candidats_taille(largeur, hauteur)
        if len(par_taille) > 12:
            reference = _vignette(falsifie)
            classes = []
            for p in par_taille:
                try:
                    classes.append((float(np.abs(index.vignette(p) - reference).mean()), p))
                except OSError:
                    continue
            par_taille = [p for _, p in sorted(classes, key=lambda x: x[0])[:12]]
        trouve = tester(par_taille, "dimensions")
    return {
        "new_id": element["new_id"],
        "source": normaliser_source(element["source_dataset"]),
        "apparie": trouve is not None,
        "original": trouve["original"] if trouve else None,
        "exif": trouve["exif"] if trouve else None,
        "voie": trouve["voie"] if trouve else None,
        "ecart_moyen": trouve["ecart_moyen"] if trouve else None,
        "part_pixels_differents": trouve["part_pixels_differents"] if trouve else None,
        "essais": len(essais),
    }


def _tache_apparier(args: tuple[dict[str, Any], str, str, str]) -> dict[str, Any]:
    element, falsifie, masque, dossier_source = args
    index = _INDEX_CACHE.get(dossier_source)
    if index is None:
        index = IndexSource(Path(dossier_source))
        _INDEX_CACHE[dossier_source] = index
    return apparier_un(element, Path(falsifie), Path(masque), index)


_INDEX_CACHE: dict[str, IndexSource] = {}


def apparier(ch: Chemins, *, processus: int = 1) -> dict[str, Any]:
    test = elements_test(lire_metadonnees(ch.metadonnees()))
    deja: dict[str, dict[str, Any]] = {}
    if ch.journal_appariement.exists():
        for ligne in ch.journal_appariement.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                r = json.loads(ligne)
                deja[r["new_id"]] = r
    a_faire = [
        (m, str(_chemin_jeu(ch, m["image"])), str(_chemin_jeu(ch, m["mask"])),
         str(ch.sources / normaliser_source(m["source_dataset"])))
        for m in test if m["new_id"] not in deja
    ]
    ch.racine.mkdir(parents=True, exist_ok=True)
    with ch.journal_appariement.open("a", encoding="utf-8") as journal:
        if processus > 1:
            # Regrouper par base garde un index par processus et par base.
            a_faire.sort(key=lambda t: t[3])
            with ProcessPoolExecutor(max_workers=processus) as pool:
                for r in pool.map(_tache_apparier, a_faire, chunksize=4):
                    journal.write(json.dumps(r, sort_keys=True) + "\n")
                    journal.flush()
                    deja[r["new_id"]] = r
        else:
            for t in a_faire:
                r = _tache_apparier(t)
                journal.write(json.dumps(r, sort_keys=True) + "\n")
                journal.flush()
                deja[r["new_id"]] = r
    return resume_appariement([deja[m["new_id"]] for m in test])


def resume_appariement(resultats: list[dict[str, Any]]) -> dict[str, Any]:
    par_source: dict[str, dict[str, int]] = {}
    for base in sorted({r["source"] for r in resultats}):
        sous = [r for r in resultats if r["source"] == base]
        par_source[base] = {
            "falsifications": len(sous),
            "apparies": sum(r["apparie"] for r in sous),
        }
    return {
        "falsifications": len(resultats),
        "apparies": sum(r["apparie"] for r in resultats),
        "voies": dict(Counter(r["voie"] for r in resultats if r["apparie"])),
        "par_source": par_source,
    }


# --------------------------------------------------------------------------
# Trajets et dossiers aveugles
# --------------------------------------------------------------------------


def appliquer_trajet(image: np.ndarray, trajet: str) -> tuple[bytes, str]:
    """Meme code, memes reglages, pour la falsification et pour l'original."""
    source = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).convert("RGB")
    tampon = io.BytesIO()
    if trajet == "natif_png":
        source.save(tampon, format="PNG")
        return tampon.getvalue(), ".png"
    if trajet == "natif_j85":
        source.save(tampon, format="JPEG", quality=QUALITE_J85)
        return tampon.getvalue(), ".jpg"
    if trajet == "demi_png":
        largeur, hauteur = source.size
        reduite = source.resize(
            (max(1, largeur // 2), max(1, hauteur // 2)), Image.Resampling.LANCZOS
        )
        reduite.save(tampon, format="PNG")
        return tampon.getvalue(), ".png"
    raise BancImpossible(f"trajet inconnu : {trajet}")


def _journal(ch: Chemins) -> dict[str, dict[str, Any]]:
    if not ch.journal_appariement.exists():
        raise BancImpossible("appariement absent : lancer l'etape `apparier`")
    table = {}
    for ligne in ch.journal_appariement.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            r = json.loads(ligne)
            table[r["new_id"]] = r
    return table


def preparer(ch: Chemins) -> dict[str, Any]:
    test = elements_test(lire_metadonnees(ch.metadonnees()))
    journal = _journal(ch)
    manquants = [m["new_id"] for m in test if m["new_id"] not in journal]
    if manquants:
        raise BancImpossible(f"appariement incomplet : {len(manquants)} elements non traites")

    # Deux populations. Le banc principal est APPARIE : chaque falsification
    # a son original, et seules ces paires entrent dans le taux de fausse
    # alerte comme dans le rappel. Les falsifications sans original prouve
    # restent mesurees a part, en rappel seul.
    documents: list[dict[str, Any]] = []
    for m in test:
        r = journal[m["new_id"]]
        commun = {
            "paire": m["new_id"],
            "source": r["source"],
            "langue": str(m.get("language") or "inconnue").lower(),
            "type_doc": str(m.get("doc_type") or "inconnu").lower(),
        }
        population = "appariee" if r["apparie"] else "non_appariee"
        documents.append({
            **commun, "verite": "ia", "population": population,
            "chemin": str(_chemin_jeu(ch, m["image"])), "exif": False,
        })
        if r["apparie"]:
            documents.append({
                **commun, "verite": "honnete", "population": population,
                "chemin": r["original"], "exif": bool(r["exif"]),
            })

    for population in ("appariee", "non_appariee"):
        sous = [d for d in documents if d["population"] == population]
        if not sous:
            continue
        ordre = list(range(len(sous)))
        random.Random(f"{GRAINE_AVEUGLE}-{population}").shuffle(ordre)
        for trajet in TRAJETS:
            dossier = ch.aveugle / population / trajet
            cle = ch.cles / population / f"{trajet}.csv"
            if cle.exists() and dossier.exists() and len(list(dossier.iterdir())) == len(sous):
                continue
            dossier.mkdir(parents=True, exist_ok=True)
            cle.parent.mkdir(parents=True, exist_ok=True)
            lignes_cle = []
            for rang, position in enumerate(ordre):
                d = sous[position]
                image = charger_rgb(Path(d["chemin"]), orientation_exif=d["exif"])
                octets, extension = appliquer_trajet(image, trajet)
                nom = f"{rang:05d}{extension}"
                (dossier / nom).write_bytes(octets)
                lignes_cle.append({
                    "fichier": nom,
                    "verite": d["verite"],
                    "format": extension.lstrip(".").replace("jpeg", "jpg"),
                    "generateur": GENERATEUR if d["verite"] == "ia" else "",
                    "source": d["source"],
                    "langue": d["langue"],
                    "type_doc": d["type_doc"],
                    "paire": d["paire"],
                })
            with cle.open("w", encoding="utf-8", newline="") as sortie:
                ecrivain = csv.DictWriter(sortie, fieldnames=list(lignes_cle[0]))
                ecrivain.writeheader()
                ecrivain.writerows(lignes_cle)
    return {
        "appariee": sum(d["population"] == "appariee" for d in documents),
        "non_appariee": sum(d["population"] == "non_appariee" for d in documents),
    }


# --------------------------------------------------------------------------
# Depouillement
# --------------------------------------------------------------------------


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(RACINE_DEPOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


# --------------------------------------------------------------------------
# Ligne de commande (release: the four steps kept)
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etape", choices=("telecharger", "sources", "apparier", "preparer", "tout"))
    parser.add_argument("--racine", type=Path, default=None, help="donnees, hors depot (AIFORGE_ROOT)")
    parser.add_argument("--processus", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ch = Chemins((args.racine or racine_par_defaut()).expanduser().resolve())
    etapes = ["telecharger", "sources", "apparier", "preparer"] if args.etape == "tout" else [args.etape]
    try:
        for etape in etapes:
            debut = time.time()
            if etape == "telecharger":
                resultat: Any = telecharger(ch)
            elif etape == "sources":
                resultat = sources(ch)
            elif etape == "apparier":
                resultat = apparier(ch, processus=args.processus)
            else:
                resultat = preparer(ch)
            print(f"[{etape}] {time.time() - debut:.0f} s : {json.dumps(resultat, ensure_ascii=False)}", flush=True)
    except BancImpossible as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
