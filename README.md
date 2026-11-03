# Beyond Zero-Shot on AIForge-Doc v2: a trained detector and a process-twin control

Code, weights, identifier lists and frozen predictions for the paper *Beyond Zero-Shot on
AIForge-Doc v2: A Trained Detector and a Process-Twin Control* (Billel Helali, 2026).

All published results on [AIForge-Doc v2](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2)
are zero-shot and close to chance (best AUC 0.599). A compact fully-convolutional network
(371,617 parameters), trained on the official TRAINING split, reaches an AUC of 0.989 on the
612 paired TEST pairs. Process twins then decompose that result. A process twin is an original
put through an approximation of the benchmark's resample-and-paste pipeline, with no change of
content; the twins are scored under a decision rule fixed before the run.

| Rebuilt network N3, native PNG, frozen threshold | Result |
|---|---|
| Forgeries flagged / originals flagged | 565/612 / 4/612 |
| AUC, forgeries against originals | 0.9894 (bootstrap 95% CI 0.9830–0.9944, over TEST pairs) |
| Twins flagged (rule fixed in advance: at most 10%) | 231/612 (37.7%; Wilson 34.0–41.7%) |
| AUC, twins against originals | 0.8801 |
| Forgery scores above its own twin | 602/612 pairs; AUC 0.9081 |
| Detections whose twin is not flagged | 336 of 565 (exact McNemar test, 336 against 2) |

Part of the signal is the benchmark's process: 231/612 twins are flagged, above the 10%
ceiling set in advance. The paper therefore reports the AUC of 0.9894 as including that
process. A substantial residual lies beyond what the twin reproduces (602/612 pairs; AUC
0.9081).

**Check it in one command:** `python verify_frozen.py` recomputes these figures from the
frozen predictions in `frozen/`, with the standard library and no download.

## Contents

| Path | What |
|---|---|
| `entrainement/aiforge_v2_detecteur_appris.py` | Crops, training, reserve scores, TEST predictions and measurement of the network |
| `entrainement/aiforge_v2_developpement.py` | TRAINING split: download and pixel pairing |
| `entrainement/banc_aiforge_v2.py` | TEST split: download, source corpora, pixel pairing, blind folders and keys |
| `entrainement/banc_maison.py` | Process twins of the TEST pairs: build and seal |
| `entrainement/mesurer_f06.py` | Two helpers: Wilson interval and file SHA-256 |
| `entrainement/runs/mesures/` | Frozen thresholds (`*_seuils_geles_*`) and the reports of the first two TEST passes (counts only) |
| `configs/banc_maison_scelle_aiforge_2026-10-29.json` | The 612 TEST pairs, sealed before the twin pass: identifier, source, origin, SHA-256 of forgery, original, mask and twin |
| `recherche/2026-11-02/aiforge/` | Twin protocol (fixed before the run), twin pass, AUC and localisation, paired analyses |
| `weights/` | The two rebuilt networks, N2 (`reseau_v2.pt`) and N3 (`reseau_v3.pt`) |
| `frozen/` | Frozen per-document predictions of the TEST pass of N3 (P3), the TEST keys, and the per-twin scores of the twin pass (P4) |
| `splits/` | Identifiers of the TRAINING pairs used for training (1,928) and of the TRAINING reserve (522) |
| `verify_frozen.py` | Recomputes the headline figures from `frozen/` |
| `SHA256SUMS` | Checksums of the weights, the lists, the sealed file and the frozen predictions |

**Identifiers.** They are the dataset's `new_id`, which is numbered within each split. The same
number in TRAINING and in TEST names two different forgeries, with different images and
specifications. TRAINING and TEST share no image identifier (`image_id`).

**Language.** Comments, docstrings, identifiers and the protocol are in French, as in the code
that produced the paper's numbers. Some docstrings describe the private repository they come
from. For example, they mention `PYTHONPATH=src`, which is not needed here, or say that no
weight is versioned. The scripts are kept verbatim so that their checksums match the ones
recorded during the runs. "Amendement 3" in comments is the internal protocol of
29 October 2026, written for another network, that set the 10% rule (paper, Section 7.3).

## Provenance

The scripts and reports come from the author's private research repository. Its commits are
named below for the record, but they are not public. The weights, the split lists and the
frozen predictions come from the data directory of the runs. `README.md`, `LICENSE`,
`CITATION.cff`, `requirements.txt`, `SHA256SUMS` and `verify_frozen.py` are new.

| File | Source commit | Change |
|---|---|---|
| `aiforge_v2_detecteur_appris.py` | `9f514b2` (29 Oct 2026) | verbatim |
| `aiforge_v2_developpement.py` | `eeff056` | verbatim |
| `banc_aiforge_v2.py` | `3fe265f` | extract: the steps that scored the blind folders with another detector of the author (`predire`, `mesurer`, report writers) are removed, with the two constants only they used; new docstring and command line; every other definition verbatim |
| `banc_maison.py` | `daa3b8b` (29 Oct 2026, the sealing commit) | extract: the AIForge twin functions, the constants they read and `SEUIL_JUMEAUX` (read by `jumeaux_v3.py`), verbatim; new docstring, import block and `main` |
| `mesurer_f06.py` | `01dca7a` | extract: `wilson`, `empreinte` and `Z_95`, verbatim |
| `jumeaux_v3.py`, `PROTOCOLE_jumeaux_v3.md` | `8290ec7` (2 Nov 2026, protocol commit) | verbatim |
| `auc_localisation_v3.py` | `52ee5d6` | one line: the repository root, an absolute path, is now `Path(__file__).resolve().parents[3]` |
| `analyses_referee_v3.py` | `d08d29b` | verbatim |
| frozen thresholds and reports | `81ee50b`, `1044f78`, `624cc47`, `aa371cd` | verbatim |
| sealed twin list | `daa3b8b` | verbatim |

**Which version ran when.**
- **Training and scoring, 29 October 2026.** The rebuilt networks were trained and scored that
  day. Two device options were committed during the run: training on an Apple GPU in `d044fdc`,
  after N2's training had started, and scoring on it in `9f514b2`, after the reserve scoring
  had started. The training code is the same in both commits.
- **Twins.** They were built the same evening, from code committed right after sealing, in
  `daa3b8b` (8 seconds after the seal's timestamp);
  the sealed list records the parent commit, `4695873`. The twin functions are identical in
  `daa3b8b`, in the later twin-pass commit `1946fd8` and in this release.
- **Twin pass, 2 November.** It ran the detector script at a later version, which adds a
  pre-trained network option that is not used here. On a synthetic page, both versions give
  the same N3 scores and the same peak box.

**What you can check without the private repository.**
- **Scoring script.** The frozen TEST predictions record the SHA-256 of the script that
  produced them, `5d9881dc917b0405…`. It is the SHA-256 of the released
  `aiforge_v2_detecteur_appris.py`; `verify_frozen.py` checks it, and checks that the
  predictions name the released N3 weights.
- **TEST keys.** The keys in `frozen/cles/` have the SHA-256 recorded as `cle_sha256` in both
  27 September reports, and step 1 regenerates them with the same seed.
- **Twins.** Six twins rebuilt from the data by the released `banc_maison.py` (three
  PNG-origin, three JPEG-origin) are byte-identical to the sealed ones. This was checked on
  macOS (Apple silicon), Python 3.13, with the versions in `requirements.txt`.

## Weights

| File | Network | SHA-256 |
|---|---|---|
| `weights/reseau_v2.pt` | N2, rebuilt on 29 Oct 2026 | `41b455bfae96d681e2b508f53c1d839696125b15be346f7daea611ea037e062f` |
| `weights/reseau_v3.pt` | N3, rebuilt on 29 Oct 2026 | `5a909a273a987838694ee643eda116a1bc9a729b700a8ec7b2acd4e3df940404` |

These are PyTorch `state_dict` files for `construire_reseau()`. They load with
`torch.load(path, map_location="cpu", weights_only=True)`. Both were trained on an Apple GPU
(MPS) with PyTorch 2.15, N2 in 3,210 s and N3 in 3,403 s.

The original weights of 27 September 2026 are lost: N2 (`8a0785ec…`, first TEST pass) and N3
(`f371c52e…`, second pass). Every AUC, localisation and twin figure of the paper comes from the
rebuilt N3. The reports in `entrainement/runs/mesures/` are those of the two original passes:
there N3 flagged 564/612 forgeries for 5/612 originals, where the rebuilt N3 flags 565 and 4.

**The rebuild differs from the original recipe in one respect.** Between the original N2
(`9629202`) and N3 (`0d63670`), the crop sampler started to double the crops on patches
smaller than 2,000 pixels, and that doubling also applies to N2's settings. So the rebuilt N2
was trained on 39,410 crops, against 34,698 for the original. The rebuilt N3 used the same
64,898 crops as the original. The released code reproduces the rebuild's recipe. Training on
an Apple GPU is not bit-reproducible, so retrained weights differ slightly.

## Reproducing

Python 3.13: run `pip install -r requirements.txt`, then run every command from the repository
root.
- **Data.** Data stay outside the repository, under `AIFORGE_ROOT`.
- **Download.** It needs a Hugging Face token in `HF_TOKEN`, from an account that has accepted
  the dataset's conditions.
- **Versions.** AIForge-Doc v2 is pinned to revision
  `1c42e42824918836255f6d9c604c9bc78ead6e08`. The source corpora of the originals (CORD,
  WildReceipt, SROIE, XFUND) are fetched at their current version. The sealed list gives the
  SHA-256 of each original, so you can check what you downloaded.

```sh
export AIFORGE_ROOT=/path/to/data HF_TOKEN=...

# 1. TEST split: download, source corpora, pixel pairing, blind folders and their keys
python entrainement/banc_aiforge_v2.py tout
# 2. TRAINING split: download and pixel pairing
python entrainement/aiforge_v2_developpement.py telecharger
python entrainement/aiforge_v2_developpement.py apparier

# 3. N2 (add --peripherique mps to train on an Apple GPU)
python entrainement/aiforge_v2_detecteur_appris.py vignettes
python entrainement/aiforge_v2_detecteur_appris.py entrainer \
  --iterations 12000 --lot 48 --poids-trajets 1,1.5,2 --modele reseau_v2.pt
# 4. N3, fine-tuned from N2
python entrainement/aiforge_v2_detecteur_appris.py vignettes \
  --vignettes vignettes_v3.npz --n-masque 3 --n-hasard 2
python entrainement/aiforge_v2_detecteur_appris.py entrainer \
  --iterations 12000 --lot 48 --vignettes vignettes_v3.npz --depart reseau_v2.pt \
  --lr 1e-3 --poids-vignette 0.5 --poids-trajets 1,1.5,2 --modele reseau_v3.pt
# 5. Optional: scores on the TRAINING reserve (writes one line per page)
python entrainement/aiforge_v2_detecteur_appris.py reserve --modele reseau_v3.pt --stats
```

**Thresholds.** The frozen thresholds (`entrainement/runs/mesures/*_seuils_geles_*.json`) were
chosen on the reserve scores of the original networks, by the rule recorded in those files.
They were then applied unchanged to the rebuilt networks. This release has no script that
applies the rule.

To use the released weights, copy `weights/reseau_v3.pt` (and `reseau_v2.pt`) into
`$AIFORGE_ROOT/dev/appris/`. The twin pass refuses any other N3 weights.

```sh
# 6. TEST pass at the frozen thresholds
#    (on an Apple GPU: APPRIS_PERIPHERIQUE=mps, with --processus 1)
python -c "import json; print(json.dumps(json.load(open('entrainement/runs/mesures/aiforge_v2_appris_v3_seuils_geles_2026-09-27.json'))['seuils']))" > seuils_v3.json
python entrainement/aiforge_v2_detecteur_appris.py predire-test --modele reseau_v3.pt --stats
python entrainement/aiforge_v2_detecteur_appris.py mesurer-test --modele reseau_v3.pt --stats --seuils seuils_v3.json

# 7. Process twins. They are written under ~/lumelia-data/banc_maison/jumeaux_aiforge
#    (constant RACINE). The sealing step refuses to overwrite the sealed list, so move it
#    aside, build, then put the original back: the twin pass checks every twin against it.
mv configs/banc_maison_scelle_aiforge_2026-10-29.json scelle_2026-10-29.json
python entrainement/banc_maison.py sceller-aiforge --processus 4
mv scelle_2026-10-29.json configs/banc_maison_scelle_aiforge_2026-10-29.json

# 8. Twin pass, then AUC and localisation of the TEST pass
python recherche/2026-11-02/aiforge/jumeaux_v3.py --sortie jumeaux-v3-resultat.json --journal jumeaux-v3.jsonl
python recherche/2026-11-02/aiforge/auc_localisation_v3.py > auc-localisation-v3.json
```

**Notes on the scripts.**
- **`auc_localisation_v3.py`.** Its docstring predates the rebuild. It reads the predictions
  of the 29 October pass, which replaced those of 27 September at the same path. Its
  fingerprint check against the 27 September report therefore reports a difference by design.
- **`analyses_referee_v3.py`.** It produced the paper's triplet-bootstrap intervals, 2x2 table
  and McNemar test; `--journal` takes the per-twin scores. It runs on macOS, because it reads
  file birth times. It also checks its numbers against the reports of the two passes, which
  are kept in the private repository with working notes and are not released.
  `verify_frozen.py` gives the same headline numbers from `frozen/`.

**General notes.**
- **Device.** Scores computed on CPU and on an Apple GPU differ by about 1e-5 on the logits.
  Weights trained on the two devices differ more.
- **Twin checksums.** The twin pass stops at the first twin whose PNG bytes differ from the
  sealed list. Another Pillow or zlib version can encode the same pixels into different bytes.
- **Twin count.** The 612 twins come from 611 distinct originals: one SROIE scan is the
  original of two pairs.
- **TEST reuse.** The paper's figures are frozen-threshold measurements on a TEST split that
  was scored several times; see its Section 9. A new test set is the next step for any
  external claim.

## Frozen predictions

`frozen/` holds the files that the paper's figures were computed from, unchanged:

- `frozen/predictions/appris_reseau_v3_stats/<population>/<path>.json`: the scores and peak
  boxes of the rebuilt N3 on the blind TEST folders (P3, 29 October 2026);
- `frozen/cles/<population>/<path>.csv`: the keys of those folders;
- `frozen/jumeaux-v3.jsonl`: the scores and peak boxes of the 612 process twins (P4,
  2 November 2026), one line per twin, in the order they were scored.

`python verify_frozen.py` recomputes the following from these files alone:
- the pages above the frozen threshold: 565, 4 and 231 of 612;
- the three AUCs: 0.9894, 0.8801 and 0.9081;
- the paired counts: 605 and 602 of 612;
- the 2x2 table and the exact McNemar test.

## Not released

The author's own forgery benches and the network of the 30 October pass. They are built
from documents that are not public.

## Licence

AIForge-Doc v2 is licensed CC BY-NC-SA 4.0. Everything in this repository is released under the
same licence, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), for
research use: code, weights, identifier lists, reports and frozen predictions. The weights are
research artefacts trained on AIForge-Doc v2 and are not part of any commercial product. See
`LICENSE`.

## Citation

```bibtex
@misc{helali2026twins,
  author = {Helali, Billel},
  title  = {Beyond Zero-Shot on {AIForge-Doc} v2: A Trained Detector and a Process-Twin Control},
  year   = {2026},
  note   = {Preprint}
}
```
