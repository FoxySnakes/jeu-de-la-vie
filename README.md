# Jeu de la Vie (Canvas Edition)

Une implémentation haute performance et esthétique du Game of Life de Conway en JavaScript pur, s'appuyant sur un Canvas HTML5 unique et des `TypedArray` pour gérer plus d'un million de cellules sans ralentissement perceptible.

## Fonctionnalités

- Simulation optimisée avec double buffer (`Uint8Array`) et gestion de l'âge des cellules.
- Rendu néon/glassmorphism avec dégradé dynamique et effet de glow.
- Zoom continu, panoramique à la souris et transitions fluides.
- Barre d'outils flottante : lecture/pause, pas à pas, remise à zéro, génération aléatoire.
- Chargement rapide de motifs célèbres (glider, pulsar, Gosper gun...).
- Panneau de paramètres (dimensions, vitesse, style visuel, densité aléatoire).
- Interface responsive utilisable sur desktop et tablette.

## Démarrage

Ouvrez simplement `index.html` dans un navigateur moderne ou servez le dossier via un serveur statique :

```bash
# Avec Python
python -m http.server 8000
# Puis rendez-vous sur http://localhost:8000
```

Aucun build ni dépendance externe n'est nécessaire.
