"""Split TRAINING d'AIForge-Doc v2 : terrain de mise au point, jamais de mesure publiee.

Le banc TEST (`banc_aiforge_v2.py`) reste fige et ne telecharge que TEST. Ce
programme recupere le split TRAINING et l'apparie aux originaux par les memes
regles pixel, pour que tout reglage (seuil, poids, choix de trait) soit fixe
ici, avant un passage unique sur TEST. Rien de ce qui sort d'ici n'est une
performance : c'est de la mise au point.

Licence : CC BY-NC-SA 4.0. Mesure et recherche seulement ; aucun poids ni seuil
issu de ce jeu n'entre dans le produit vendu.

    PYTHONPATH=src python entrainement/aiforge_v2_developpement.py telecharger
    PYTHONPATH=src python entrainement/aiforge_v2_developpement.py apparier
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import banc_aiforge_v2 as banc


def elements(ch: banc.Chemins, split: str) -> list[dict[str, Any]]:
    lignes = banc.lire_metadonnees(ch.metadonnees())
    choisis = [m for m in lignes if str(m.get("split", "")).lower() == split]
    if not choisis:
        raise banc.BancImpossible(f"aucune ligne split={split}")
    return sorted(choisis, key=lambda m: m["new_id"])


def journal(ch: banc.Chemins, split: str) -> Path:
    return ch.racine / f"appariement_{split}.jsonl"


def lire_journal(ch: banc.Chemins, split: str) -> dict[str, dict[str, Any]]:
    if split == "testing":
        chemin = ch.journal_appariement
    else:
        chemin = journal(ch, split)
    table: dict[str, dict[str, Any]] = {}
    if chemin.exists():
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                r = json.loads(ligne)
                table[r["new_id"]] = r
    return table


def telecharger(ch: banc.Chemins, *, fils: int = 8) -> dict[str, int]:
    entetes = {"Authorization": f"Bearer {banc._jeton()}"}
    train = elements(ch, "training")
    chemins = [p for m in train for p in (m["image"], m["mask"])]

    def un(chemin: str) -> Path:
        relatif = chemin.split("AIForge-Doc-v2.0/")[-1].lstrip("/")
        return banc._telecharger_url(
            banc._url_hf(banc.DEPOT_HF, relatif, banc.REVISION_HF), ch.jeu / relatif, entetes=entetes
        )

    with ThreadPoolExecutor(max_workers=fils) as pool:
        list(pool.map(un, chemins))
    return {"elements_training": len(train), "fichiers": len(chemins)}


def apparier(ch: banc.Chemins, *, processus: int = 1) -> dict[str, Any]:
    train = elements(ch, "training")
    deja = lire_journal(ch, "training")
    a_faire = [
        (m, str(banc._chemin_jeu(ch, m["image"])), str(banc._chemin_jeu(ch, m["mask"])),
         str(ch.sources / banc.normaliser_source(m["source_dataset"])))
        for m in train if m["new_id"] not in deja
    ]
    # Bases JPEG d'abord : ce sont elles que les tests de grille regardent.
    a_faire.sort(key=lambda t: (t[3].endswith("cord"), t[3]))
    with (
        journal(ch, "training").open("a", encoding="utf-8") as sortie,
        ProcessPoolExecutor(max_workers=max(1, processus)) as pool,
    ):
        for r in pool.map(banc._tache_apparier, a_faire, chunksize=4):
            sortie.write(json.dumps(r, sort_keys=True) + "\n")
            sortie.flush()
            deja[r["new_id"]] = r
    return banc.resume_appariement([deja[m["new_id"]] for m in train])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etape", choices=("telecharger", "apparier"))
    parser.add_argument("--racine", type=Path, default=None)
    parser.add_argument("--processus", type=int, default=4)
    args = parser.parse_args(argv)
    ch = banc.Chemins((args.racine or banc.racine_par_defaut()).expanduser().resolve())
    try:
        resultat = telecharger(ch) if args.etape == "telecharger" else apparier(ch, processus=args.processus)
    except banc.BancImpossible as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2
    print(json.dumps(resultat, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
