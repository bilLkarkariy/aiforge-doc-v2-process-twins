# Article AIForge-Doc v2 : le contrôle des jumeaux de la v3

Protocole écrit le 02/11/2026, avant qu'aucun jumeau ne soit noté par la v3.

## La question

Le détecteur appris v3 a été entraîné sur le jeu d'entraînement d'AIForge-Doc v2. Au troisième passage annoncé sur TEST (29/10), il détecte 565 faux sur 612 et accuse 4 originaux sur 612. Son AUC est de 0,989 (`52ee5d6`).

Voit-il la retouche de GPT-Image-2, ou le procédé qui l'entoure ? Ce procédé, c'est la fenêtre rééchantillonnée, puis recollée.

## La population

On prend les 612 jumeaux de collage scellés le 29/10 (commit `daa3b8b`, scellé de l'amendement 3). Chaque jumeau est fabriqué ainsi :
- on part de l'original décodé ;
- on fait un aller-retour LANCZOS vers 1024 × 1024 dans la fenêtre du procédé ;
- on recolle le résultat sur le rectangle du masque, plus 6 px de marge, sans changer aucune valeur.

L'empreinte de chaque jumeau est relue contre le scellé avant toute note. Une seule empreinte changée arrête le passage.

## Le réseau et le seuil

- Le réseau est `reseau_v3.pt`, reconstruit le 29/10 (empreinte `5a909a27…`).
- Le seuil est celui du trajet PNG natif, gelé sur la réserve d'entraînement : `haut16` ≥ 6,5694.
- Rien n'est réglé pour ce passage.

## Ce qui est mesuré

- Les jumeaux accusés au seuil gelé, sur 612, avec l'intervalle de Wilson à 95 %. Le compte est donné par origine : PNG (198) et JPEG (414).
- Pour chaque jumeau accusé, si la boîte du pic touche le masque.
- L'AUC des faux contre les jumeaux, et celle des jumeaux contre les originaux.
- Le nombre de paires où le faux obtient un score plus haut que son jumeau.

## La règle de décision, fixée avant le passage

La règle est celle de l'amendement 3 (`SEUIL_JUMEAUX` = 0,10), reprise telle quelle.

| Jumeaux accusés | Ce que l'article dira |
|---|---|
| 10 % ou moins | La v3 voit la retouche au-delà du procédé. |
| Plus de 10 % | La v3 voit le procédé. Le 0,989 et les 565 faux détectés seront présentés comme un raccourci du banc, pas comme une détection de la retouche. |

Les deux résultats seront publiés tels quels.

## Les garde-fous

- Le passage est unique.
- Un passage perdu, par exemple sur un plantage, est refait en entier sur les mêmes fichiers. Il n'est jamais filtré.

## Les limites déclarées

- Le jumeau approche le procédé d'AIForge ; le pipeline exact du jeu n'est pas publié à ce niveau de détail.
- Le réseau noté est celui reconstruit le 29/10, pas celui du 27/09.
