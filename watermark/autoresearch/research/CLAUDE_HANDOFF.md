# Prompts autonomes pour Claude Code

Préparés le 21/11/2026. Recherche complète : [SOLUTION_LANDSCAPE.md](SOLUTION_LANDSCAPE.md). Spécifications : [BRANCH_SPECS.md](BRANCH_SPECS.md).

Copier un bloc entier pour développer une branche. Les consignes communes sont répétées pour que chaque bloc soit autonome. Commencer par E01, E03 et l'inspection E04 ; les autres dépendent des diagnostics et du budget. Ce document n'autorise aucune publication de résultats au leaderboard.

## E01 — Préserver les activations signées

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e01-signed-features. Catégorie A : seul watermark/autoresearch/train.py peut modifier le pipeline ; aucun nouveau cache ni paquet.

Hypothèse : log1p(clamp(x,min=0)) élimine une information utile. Commence par mesurer sur les cartes des images train la fraction de valeurs négatives et les ranges. Si toutes les cartes sont non négatives, documente ce résultat et n'invente pas une amélioration.

Ajoute une sélection current/signed/identity dans inputs(). signed(x)=sign(x)*log1p(abs(x)). Ne modifie ni RGB, ni accord, ni occlusion, ni tête. Teste d'abord une transformation à la fois ; journalise par méthode/carte sans ajuster de statistiques sur validation.

Tests : égalité current/signed pour x≥0, signe conservé pour x<0, absence NaN/Inf, formes inchangées, gradients tête présents et extracteurs figés. Entraîner la référence et la variante avec mêmes seeds/folds/seuils. L'échec scientifique est l'absence de gain reproductible ; la réussite technique est une comparaison valide, même négative.

Motivation : audit du code, pas une méthode publiée. Un gain uniquement dû à une sélection de seuil différente ne démontre pas l'hypothèse. Livrer les deltas appariés par méthode, la proportion d'activations affectées et une décision explicite. Coût attendu relatif faible, à mesurer.
```

## E02 — Pyramide et contrôle du RGB

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e02-native-pyramid. Catégorie A : modifier uniquement train.py, sans nouveaux caches.

Hypothèse : la fusion progressive des cartes disponibles préserve mieux les petits objets et trous. Lire les shapes réelles puis implémenter projections 1×1 et fusion compacte aux grilles 32→64→128. Proposition initiale : projections 16 canaux, largeur 32. Redimensionner explicitement chaque RGB/feature à son point de fusion ; pas d'attention lourde.

Conserver loss et message comme dans la référence. La première variante remplace seulement la tête. La seconde désactive RGB sur cette même tête. Comparer aussi paramètres/mémoire à la référence pour distinguer capacité et fusion. Ne présenter aucune interpolation comme une nouvelle observation haute résolution.

Tests : shapes sur les trois méthodes, sorties [B,128,128], gradient, contrôles sans RGB, tiny-overfit sur train. Mesurer mIoU des petites régions et trous avec effectifs. Réfutation si aucun gain ou si seul un changement de capacité explique le résultat. Pas de suppression systématique de petites composantes.

Référence conceptuelle : TruFor §3.2 https://arxiv.org/html/2212.10957v2 ; nous adaptons l'idée de fusion, sans importer ses poids ni revendiquer ses résultats. Documenter coût relatif faible–moyen et risque de raccourcis sémantiques.
```

## E03 — Preuve locale par bit

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e03-message-evidence. Catégorie B. Créer watermark/autoresearch/experiments/E03/{prepare_evidence.py,train_evidence.py} ; ne pas modifier les extracteurs/harness partagés. Cache séparé ar_cache_research/E03/<manifest_hash>.

Hypothèse : les distributions d'accord par bit sont plus informatives que les deux moyennes existantes. PixelSeal et TrustMark seulement au MVP ; StegaStamp demeure identique à la référence.

Inspecter features() des wrappers et le format réel des messages. Extraire local_logits [N,K,h,w], message [N,K], noms et provenance ; K/h/w vérifiés, pas imposés. Garder la densification de référence. Les messages complets ne sont pas dans le cache actuel : lire les métadonnées via le loader.

Calculer r=(2m-1)*logits ; proposer moyenne, dispersion, quantile bas, fraction positive, Z=sum(r)/sqrt(sum(logits²)+eps). Ajouter ces cartes à la tête existante ; commencer par quelques statistiques, avant projection apprise par bit. Comparer à moyenne seule avec capacité appariée.

Tests : parité des accords reconstruits avec le cache, identité noms, vrai message versus messages de contrôle à poids fixes, cas logits nuls. Les messages ECC peuvent être corrélés : Z n'est pas une p-value de présence. Vérifier les redondances algébriques : vrai/faux message avec un simple score linéaire peut ne faire que renormaliser l'accord existant. Ne pas entraîner avec cible tout-zéro pour un mauvais message : notre label concerne la présence physique du watermark.

Motivation : Pixel Seal §3.1 https://arxiv.org/html/2512.16874v1 et WAM §3 https://arxiv.org/html/2411.07231v2. Succès : nouveau signal spécifique au message et gain OOF reproductible ; sinon rapport négatif. Mesurer coût et volume avant extraction de tout le test.
```

## E04 — Alignement StegaStamp

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e04-stega-coordinates. Catégorie B conditionnelle. Créer experiments/E04/{inspect_geometry.py,prepare_geometry.py,train_geometry.py} et une copie privée du graphe ONNX ; ne pas modifier l'original.

Hypothèse à vérifier AVANT entraînement : les cartes post-STN pourraient ne pas être alignées avec le masque original. Le code officiel applique une transformation spatiale avant les convolutions, mais l'export du challenge peut différer. Sources : https://raw.githubusercontent.com/tancik/StegaStamp/master/models.py et StegaStamp §4.2 https://conferences.computer.org/cvpr/pdfs/CVPR2020-1XMljIyuXWg2zC9bnL19Tw/716800c114/716800c114.pdf

Inspecter le graphe, les axes, la transformation et les tenseurs réellement exposables. Si le STN est absent, identité ou déjà compensé, livrer le diagnostic et ne pas forcer une solution. Sinon conserver la transformation, reprojeter les cartes dans le repère d'entrée avec un canal de validité. Aucun nom de theta ni forme [N,2,3] supposé avant inspection.

Tests : logits identiques entre graphe original et copie enrichie ; motifs de coordonnées ; identité, translation connue, aller-retour et hors-champ. Déterminer si la grille réalise sortie→entrée ou l'inverse ; valider numériquement les conventions avant grid_sample. Les trous de reprojection ne sont pas des labels propres.

Comparer mêmes cartes/tête/loss/seuils avec et sans reprojection. PixelSeal et TrustMark inchangés. Message utilisé via scalaires/occlusion existants. Rapport mIoU versus amplitude de transformation. Abandonner si géométrie correcte et aucune amélioration. Coût moyen conditionnel ; succès technique = repère démontré correct, pas promesse de gain.
```

## E05 — Résidus natifs multibandes

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e05-native-residuals. Catégorie B. Créer experiments/E05/{prepare_residuals.py,train_residuals.py}, caches distincts et manifeste ; aucun modèle externe.

Hypothèse : le pooling RGB64 fait perdre des traces utiles. Lire l'image 512 avant toute réduction. Avec torch, calculer x-G_sigma(x) pour sigma=1,2,4 pixels et des dérivées horizontales/verticales. Résumer chaque réponse par moyenne signée et RMS sur blocs 4×4 pour une grille 128 ; documenter la forme exacte des canaux. Ne pas transformer le masque par ces filtres.

Ajouter une petite branche résiduelle (projection 16 canaux comme départ) en fusion tardive avec la tête actuelle. Paramètres de normalisation estimés seulement sur train-fold et réutilisés au test. Message via les canaux existants. Même loss/epochs/folds.

Contrôles : baseline, ajout résidus, gradients RGB seuls, contrôle de capacité, permutation des résidus entre images de même méthode comme diagnostic. Examiner textures uniformes/détaillées et petites régions. Tester finitude, image constante, frontières de padding, gradients et alignement des noms.

Motivation : séparation RGB/indices faibles de TruFor https://arxiv.org/html/2212.10957v2 ; ceci ne réimplémente pas Noiseprint++. Ne pas affirmer que tout watermark est haute fréquence. Succès : gain sur vraies images OOF non réduit aux contours ; sinon abandon. Mesurer coût extraction/train et ne pas extraire tout le test avant validation du candidat.
```

## E06 — Interventions contrôlées par le message

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e06-counterfactual-probes. Catégorie B. Créer experiments/E06/{prepare_probes.py,train_probes.py}, conserver prepare_occ.py et ses caches.

Hypothèse : la distribution de l'effet d'occlusion sur vrai/faux messages sépare mieux le watermark de la simple sensibilité au flou. La baseline utilise déjà del8, ne pas la redécouvrir.

MVP sur PixelSeal/StegaStamp : conserver les 64 cases d'occlusion et le flou existants, mais sauvegarder les logits de l'image et des interventions dans [N,65,K], avec positions, noms et messages. Comparer score brut mean((2m-1)*logits) et score normalisé actuel. Calculer delta vrai message, dispersion des deltas de messages de contrôle et delta standardisé. Ces messages réutilisent les logits : aucun appel modèle supplémentaire requis.

Ajouter les cartes à tête constante. Référence inclut del8 ; comparer aussi la variante brute+standardisée déjà préparée en pending/e33_occ_raw.py. Tester une autre intervention (moyenne locale) seulement en deuxième étape, à nombre de passes égal. Ne pas supposer qu'un flou retire tout watermark ou que sa variation est toujours positive.

Contrôles : mauvais messages valides si ECC, positions aléatoires, énergie du blur, cas carte constante, calcul AUROC avec rangs moyens des ex æquo ; la décision repose sur mIoU. Moyenne sous signes aléatoires peut être nulle : ne pas vendre une soustraction nulle comme nouveau mécanisme.

Références : RISE https://cse.iitkgp.ac.in/~adas/papers/BMVC_2018_RISE.pdf et sanity checks https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf. Mesurer le provider ONNX réel et le coût sur petit lot. L'extraction ne doit pas être tuée par le worker 600 s ni étendue silencieusement à une grille plus fine.
```

## E07 — Adapter légèrement le décodeur

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e07-local-adaptation. Catégorie B, hors boucle fixe train.py seul. Créer experiments/E07/train_adaptation.py et un adaptateur différentiable privé, sans modifier les wrappers partagés.

Hypothèse : un petit ajustement supervisé spatialement peut améliorer les cartes d'un décodeur global. PixelSeal/TrustMark au départ ; StegaStamp ONNX exclu faute de chemin différentiable vérifié.

Inspecter les modules et prétraitements. Démontrer la parité des features/logits avec le wrapper figé avant changement. Ne pas appeler FrozenExtractor.__call__ sous no_grad pour entraîner. Dégeler le dernier bloc seulement, garder les autres poids gelés et les BatchNorm en eval. Tête de référence et segmentation BCE+Dice seule au MVP ; LR extracteur proposé à 1/10 du LR tête.

Repartir des poids officiels à chaque fold. Aucun cache global après entraînement sur toutes les images ; tout cache appris est propre au fold. Même nombre de vues que la référence, ou référence congelée explicitement appariée si réduction des vues imposée par le coût.

Extension uniquement après résultat du MVP : supervision auxiliaire locale de bits, sur régions marquées, sans donner au réseau une voie lui permettant de recopier le message en entrée. Motiver et vérifier les cibles à résolution grossière ; elles ne sont pas une preuve de présence dans tout le champ réceptif.

Tests : parité numérique, gradients seulement dans modules autorisés, isolation des folds, mini-overfit et validation réelle. Comparer gain train/validation, coût, mémoire. Source de l'idée de supervision locale : WAM §4.2 https://arxiv.org/html/2411.07231v2. Aucun transfert garanti. Si plus cher et sans gain confirmé, conserver le modèle figé.
```

## E08 — Synthèse à contenu identique

```text
Tu travailles sur la localisation de watermark : images 512×512, masques binaires 128×128, 510 images annotées, méthodes PixelSeal/TrustMark/StegaStamp et messages connus au test. Score officiel : moyenne des IoU 0 et 1 par image puis moyenne des images ; union vide = 1.

Lis les instructions AGENTS/CLAUDE/TEAM_GUIDE applicables. Utilise comme point de comparaison le commit 26e5e0db4b6b34139c6d2396958a26fa22e280d5 ; si absent ou si le projet a évolué, identifie explicitement l'écart. Crée une branche et un worktree isolés sans déplacer le checkout de l'utilisateur ni effacer ses changements. Réutilise les folds du cache et prepare.load_cache/evaluate_oof/write_submission. Les wrappers réels sont à inspecter ; aucune API ni forme de canaux supposée.

La référence a une tête par méthode, BCE+Dice, 60 epochs, 3 folds et 3 seeds, des cartes d'accord, des sidecars del8 facultatifs et deux seuils selon bit accuracy>0,75. Fixe explicitement la présence des sidecars pour la comparaison. Conserve folds, ordre des noms, loss et hyperparamètres sauf changement spécifiquement demandé. Aucun label de validation dans l'entraînement, les statistiques apprises ou les prototypes. full_watermark n'est pas une entrée au test.

Produis code/configuration, manifeste des poids/caches/prétraitements, smoke test et commandes d'exécution. Pour toute évaluation autorisée : probabilités OOF, métriques officielles globales/par méthode/par couverture, scores par image, seuils, temps et mémoire. Comparer candidat et référence avec mêmes seeds et budget ; seuil fixe 0,5 plus calibration exploratoire identique. Ne qualifier de confirmé qu'un gain vérifié avec calibration interne au train-fold et analyse appariée. Aucun résultat inventé.

Prépare le travail sans lancer de job tant que l'état de toute l'équipe, le budget et les assets ne sont pas vérifiés. Un seul job Slurm par équipe, en cours ou en attente ; jamais un job par branche. Pas de calcul lourd sur login, d'installation dans l'environnement partagé, d'écrasement de cache ni de publication leaderboard. Si l'exécution reste indisponible, livre les scripts et vérifications réalisables en indiquant ce qui n'a pas tourné.

Branche : research/wm-e08-matched-synthesis. Catégorie C conditionnelle. Créer experiments/E08/{generate_matched.py,train_synthetic.py} et un registre de provenance, sans modifier les datasets partagés.

Hypothèse : des paires propres/marquées à contenu identique réduisent les raccourcis sémantiques. Avant toute génération : vérifier règles du challenge, autorisation des sources et embedders, poids et format de message. Si ces informations manquent, livrer le plan/code indépendant et les prérequis manquants ; ne pas télécharger un corpus ni utiliser le test.

Construire une source réellement propre x et sa version w watermarkée par la méthode exacte. Former x_mix=M*w+(1-M)*x, avec masques irréguliers, petites zones et trous. Masque à 512 puis downsampling officiel à 128. Les sources et toutes leurs variantes restent dans le train du fold correspondant ; pas de mélange avec la validation.

Vérifier l'API embed réelle et l'éventuel codage ECC. Changer le message entre variantes. Ne pas traiter une image floutée ou nettoyée par un modèle comme propre par définition. Si seule une région propre du train est disponible, signaler le décalage de domaine du crop.

MVP : préentraînement synthétique de la tête puis entraînement réel, avec baseline réel-répété au même nombre d'étapes. Même loss et extracteurs figés au départ. Contrôle propre/propre avec coutures identiques et masque nul ; évaluation sur vraies images OOF uniquement. Sauver source ID, fold, seed, message, masque, versions/forces de l'embedder.

Motivation : composition WAM §4.2 et annexe D.2 https://arxiv.org/html/2411.07231v2. Ceci ne garantit pas la même distribution que les retouches du challenge. Succès = gain sur réel, pas score synthétique. Échec si apprentissage des coutures, fuite de source ou incompatibilité des règles. Aucune génération massive avant pilote concluant.
```


