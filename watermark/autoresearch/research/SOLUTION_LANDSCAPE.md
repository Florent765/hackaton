# Localisation de watermark : paysage des solutions

Audit documentaire du 21 novembre 2026. Recherche et conception uniquement : aucun entraînement, aucun changement du pipeline, aucune connexion au serveur, aucune soumission.

## 1. Ce qui est établi, et avec quel niveau de preuve

- **CODE** : comportement lisible dans les fichiers locaux, sans preuve de fonctionnement sur les données distantes.
- **JOURNAL** : résultat consigné dans `../results.tsv` ; logs bruts, checkpoints et prédictions non disponibles ici, donc non reproduit.
- **SOURCE** : affirmation soutenue par une publication ou un code officiel consulté.
- **HYPOTHÈSE** : proposition de cette revue, à confirmer expérimentalement. Aucun gain promis.

Référence locale observée : commit `26e5e0db4b6b34139c6d2396958a26fa22e280d5`. SHA-256 de `../train.py` : `56F0015F1B48E4B644CF61178317F2CFFD151FC0B8DFCF75BADB967841ADCE77` ; de `../prepare.py` : `017F7088FDB1867AC9BC0B7B0044C42BAC1941C740178D738A8EBF7AF9DDF4F7` ; de `../../baseline.py` : `E85A540CBCD377EB3A66E5437AB33CAEC6CFA85346C6A66BB4FF3F6DD0334689`.

Instructions consultées : `../../../CLAUDE.md`, `../../../TEAM_GUIDE.md`, `../program.md`. Les consignes AGENTS fournies dans la conversation demandent de distinguer faits et hypothèses. Aucun autre AGENTS.md trouvé dans l'arborescence examinée. Les anciens prompts constituent des documents de contexte, pas une autorisation de lancer des jobs ou de publier.

### Contrat

Le PDF local [Watermark Localization](<C:/Users/mathi/Downloads/Watermark Localization.pdf>), lu intégralement dans cette conversation, décrit 510 images train, 1 020 test, trois méthodes également représentées, et des images 512 × 512. Méthode et message sont connus au test. Le masque final 128 × 128 indique la présence du watermark, pas la présence d'une retouche. Chaque cellule cible correspond à au moins 8 pixels watermarkés sur 16. Score : moyenne des deux IoU par image puis moyenne des images ; union vide = 1. Soumission : noms uniques, masque binaire, `.pt` ≤ 20 MB. Les formes peuvent être trouées ou irrégulières.

`prepare.evaluate_oof()` délègue à `wmloc.utils.miou_image`. `write_submission()` réordonne selon les métadonnées test et contrôle forme/type/taille. **Limite CODE** : conversion en bool avant contrôle et absence de contrôle explicite des noms supplémentaires. Valider valeurs binaires et égalité exacte des ensembles de noms en amont.

Les wrappers `wmloc`, les données, caches et poids n'existent pas dans le checkout inspecté. Les deux liens Hugging Face indiqués par le PDF et le plan n'ont pas été accessibles via l'outil web ; cela ne prouve pas leur indisponibilité générale. Le contrat du loader n'a donc pas été vérifié directement. Pas d'identité supposée entre les copies locales et JURECA.

### Pipeline actuel

| Élément | Observation CODE | Limite |
| --- | --- | --- |
| Extraction | `FrozenExtractor`, poids torch gelés, eval, sans gradient | Versions et poids distants non vérifiés |
| PixelSeal | `embedding`, `convnext`, `per_position_logits` | Canaux à mesurer ; 8 × 8 avant densification selon documentation locale |
| TrustMark | `layer2`, `layer3`, `per_position_logits` | 31 × 31, 16 × 16, 8 × 8 annoncés localement, non mesurés ici |
| StegaStamp | Copie ONNX exposant `dec_conv_3/Relu:0` et `dec_conv_5/Relu:0` | Ordre des axes et géométrie du graphe à vérifier ; 100 × 100 et 50 × 50 annoncés |
| Accord local | Deux canaux : moyenne des logits signés par le message et moyenne des accords binaires signés | Les logits complets par bit ne sont pas conservés |
| Contexte | RGB `[N,3,64,64]`, scalaires `[N,2]` : accord normalisé et bit accuracy | RGB basse résolution ne conserve pas les résidus fins |
| Densification | 16 passes via translations circulaires, entrelacement ×4 des seules cartes ayant la taille des logits locaux | Plus d'échantillons, pas de réduction du champ réceptif |
| Cache principal | `feats`, `names`, `method`, `feature_names`, `shifts`; train : `masks`, `full_watermark`, `fold` | Ni RGB 512, ni messages complets, ni logits par bit ; carte spatiale stockée en float16 |
| Occlusion | Sidecars facultatifs `{split}_{method}_occ.pt`, `del8` `[N,1,8,8]`, PixelSeal et StegaStamp | Chargés seulement si présents : un même code peut changer de comportement selon les fichiers |
| Tête | Une par méthode, projections 16 canaux, largeur 64, dilatations 1/2/4/8, grille 64 puis logits 128 | BCE + Dice deux classes, 60 epochs, AdamW/OneCycle |
| Validation | 3 folds fixés, stratifiés méthode × `full_watermark`, 3 seeds par fold | 27 entraînements ; score exploratoire avec réglage de seuils sur l'OOF |
| Seuils | Deux seuils par méthode, séparation par bit accuracy > 0,75 | 13 × 13 couples évalués, contrairement au protocole écrit « un seuil » |

L'occlusion remplace une case 64 × 64 par du flou et mesure la variation d'accord global. `train.py` standardise chaque carte par image, perdant son niveau absolu. `pending/e33_occ_raw.py` traite déjà cette question : ne pas la présenter comme une découverte nouvelle.

### Résultats disponibles et incohérences

**JOURNAL** : baseline 0,499778 ; trois seeds 0,501068 ; log1p 0,503177 ; occlusion 0,504098 ; deux seuils 0,507876. Le journal rapporte aussi des essais négatifs : Lovász+BCE, lissage, plus d'epochs, dropout, contexte global. Ce sont des points observés, pas des preuves d'inutilité générale. Les résultats ne sont pas comparables si caches, seuils ou folds diffèrent.

- `TASK_PLAN.md` annonce encore que l'implémentation n'a pas commencé : périmé.
- `MEGA_PROMPT.md` décrit un historique antérieur et affirme une couverture bimodale ; vérifier sur les masques avant d'en faire une hypothèse structurelle.
- Le test e31 change les folds, sans baseline appariée sur ces nouveaux folds dans le journal : le score seul ne confirme pas le gain.
- `analyze_oof.py` attend un seuil scalaire ; le fichier OOF actuel enregistre une paire. Sa comparaison directe n'est plus compatible. Adapter uniquement une copie d'analyse.
- Le calcul AUROC du probe utilise des rangs sans moyenne des ex æquo : risque de diagnostic erroné. Une carte constante doit produire 0,5 quand les deux classes existent.
- « Noms hachés et messages distincts » ne prouve pas l'absence de sources partagées. On peut rechercher des quasi-doublons visuels sans garantir leur identification.
- `TIME_BUDGET=300` n'est pas appliqué par `train.py` ; le worker impose 600 s. Les extractions coûteuses nécessitent un budget distinct.
- Les caches sont réutilisés sur simple présence : pas de manifeste de version. Les branches doivent imposer leurs propres contrôles de provenance.
- L'affirmation du commentaire selon laquelle le wrap-around n'affecte que les dernières cellules n'est pas démontrée : son influence peut se propager par le champ réceptif.

## 2. Sources scientifiques et transferts justifiés

Les liens ci-dessous ont été consultés ; les résumés décrivent leurs mécanismes, pas une reproduction de leurs expériences. Les pistes de la section 3 sont nos adaptations proposées.

### S1 — Pixel Seal

[Souček et al., version v1, §§3–5](https://arxiv.org/html/2512.16874v1), [code officiel VideoSeal](https://github.com/facebookresearch/videoseal).

**SOURCE** : entraînement de l'embedder avec objectifs adversarial et message ; extracteur ConvNeXt-v2 Tiny ; adaptation de résolution et atténuation JND. La détection par distance au message repose sur des hypothèses de distribution des bits sous absence de watermark. Les expériences portent surtout sur robustesse et imperceptibilité, pas notre masque après retouche. **Transfert proposé** : conserver la preuve par bit et contrôler l'effet des textures/JND. Un score élevé sur un emplacement dont le champ réceptif est large ne localise pas nécessairement ce pixel. Ne pas supposer une bande de fréquences universelle.

### S2 — TrustMark

[Bui et al., prépublication v1, §§3.1–3.3 et annexes](https://arxiv.org/html/2311.18297), [publication ICCV signalée par Adobe](https://research.adobe.com/publication/trustmark-robust-watermarking-and-watermark-removal-for-arbitrary-resolution-images/), [code officiel](https://github.com/adobe/trustmark), [implémentation des variantes](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/trustmark.py).

**SOURCE** : le texte consulté décrit un extracteur ResNet50, une adaptation par résidu redimensionné et un réseau de retrait. Le code identifie Q comme variante ResNet50. Le PDF ICCV a renvoyé 403 ; ne pas prétendre avoir vérifié l'identité entre toutes ses versions et la prépublication. **Transfert proposé** : cartes intermédiaires, contrôle de l'échelle et comparaison avant/après retrait. Ni le retrait ni le blur ne garantissent une image propre pixel par pixel.

### S3 — StegaStamp

[Tancik, Mildenhall, Ng, CVPR 2020, §§3–5](https://conferences.computer.org/cvpr/pdfs/CVPR2020-1XMljIyuXWg2zC9bnL19Tw/716800c114/716800c114.pdf), [code du décodeur](https://raw.githubusercontent.com/tancik/StegaStamp/master/models.py), [dépôt](https://github.com/tancik/StegaStamp).

**SOURCE** : décodeur avec transformation spatiale, convolutions puis couches denses. Le détecteur de photographies du papier est une étape distincte de la récupération du message. **Déduction** : les cartes après transformation peuvent être dans un autre repère que le masque. La présence exacte de cette opération dans l'export du challenge reste à vérifier. Le code TensorFlow officiel n'est pas une conversion PyTorch fournie par notre projet. La copie ONNX locale ajoute des sorties ; elle ne rend pas le graphe différentiable dans PyTorch.

### S4 — Watermark Anything

[Sander et al., ICLR 2025, v2, §§3–5, annexe D](https://arxiv.org/html/2411.07231v2), [code officiel](https://github.com/facebookresearch/watermark-anything).

**SOURCE** : détection et décodage par pixel, entraînement conjoint avec mélange spatial d'une image et de sa version watermarkée. La loss de bits porte sur les régions marquées. **Transfert proposé** : supervision locale auxiliaire et génération de paires à contenu identique. Le modèle publié est entraîné pour son propre watermark : l'utiliser directement sur les trois méthodes serait un transfert non validé, pas une solution prête à l'emploi. Les benchmarks WAM ne sont pas ceux du challenge.

### S5 — TruFor

[Guillaro et al., CVPR 2023, §§3.1–3.3 et annexe A](https://arxiv.org/html/2212.10957v2), [code et disponibilité](https://github.com/grip-unina/TruFor).

**SOURCE** : fusion de RGB et d'une empreinte de bruit apprise ; sorties de localisation et de confiance. **Transfert proposé** : séparer indices sémantiques et faibles traces résiduelles, puis fusionner. Les traces de manipulation ou de caméra ne sont pas des labels de watermark. La photographie synthétique peut changer la distribution des empreintes. Les poids externes, conditions d'utilisation et règles du challenge doivent être vérifiés avant importation ; la piste avec filtres fixes n'en dépend pas.

### S6 — RISE

[Petsiuk, Das, Saenko, BMVC 2018, §3 et évaluations insertion/suppression](https://cse.iitkgp.ac.in/~adas/papers/BMVC_2018_RISE.pdf).

**SOURCE** : importance estimée à partir de masques aléatoires et du score du modèle interrogé, sans gradients. **Transfert proposé** : remplacer le score de classe par une compatibilité avec le message et comparer des interventions contrôlées. Notre différentiel vrai/faux message et nos baselines de flou constituent une adaptation, pas le protocole original. La redondance d'un watermark peut rendre une suppression locale peu informative malgré une vraie présence.

### S7 — Sanity Checks for Saliency Maps

[Adebayo et al., NeurIPS 2018, tests de randomisation](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf).

**SOURCE** : certaines cartes visuellement convaincantes sont peu sensibles aux paramètres appris. **Transfert proposé** : contrôler le message, les poids et les contours RGB avant de conclure qu'une saillance révèle le watermark. Passer ces contrôles ne prouve pas une localisation correcte ; la mIoU sur masques reste nécessaire.

### S8 — Lovász

[Berman, Rannen Triki, Blaschko, CVPR 2018, §2 et annexe](https://arxiv.org/pdf/1705.08790), [implémentation officielle PyTorch](https://raw.githubusercontent.com/bermanmaxim/LovaszSoftmax/master/pytorch/lovasz_losses.py).

**SOURCE** : surrogate différentiable de la perte Jaccard ; distinction entre calcul par image et agrégation globale. **Déduction pour nous** : un essai doit traiter les deux classes et la convention des masques vides. Un remplacement générique a déjà été essayé dans le journal : priorité faible sans nouveau diagnostic. Une loss mieux alignée ne recrée pas un signal absent des entrées.

## 3. Catalogue de 18 expériences

Toutes les propositions ci-dessous sont des **HYPOTHÈSES**. A = `train.py` et caches actuels seulement ; B = nouvelles extractions/chemin d'entraînement, mêmes données et poids autorisés ; C = données ou modèles supplémentaires/règles à vérifier. Coûts relatifs, non mesurés. Les branches détaillées sont dans [BRANCH_SPECS.md](BRANCH_SPECS.md).

| ID | Idée | Mécanisme | Cat. | Coût | Risque | Priorité |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | Préserver le signe des features | Éviter l'écrêtage d'information | A | Faible | Faible | P1, branche |
| E02 | Pyramide avec RGB facultatif | Garder détails et contexte sans raccourci RGB | A | Faible–moyen | Moyen | P2, branche |
| E03 | Preuve par bit et mauvais messages | Spécificité au message plutôt que moyenne seule | B | Moyen | Moyen | P1, branche |
| E04 | Repère StegaStamp | Reprojeter les cartes transformées | B | Moyen | Moyen | P1, branche conditionnelle |
| E05 | Résidus multibandes natifs | Préserver les traces avant réduction | B | Moyen | Moyen | P2, branche |
| E06 | Interventions différentielles | Isoler l'effet lié au message | B | Élevé en extraction | Élevé | P2, branche |
| E07 | Adaptation légère supervisée | Apprendre à localiser dans l'extracteur | B | Élevé | Élevé | P3, branche |
| E08 | Synthèse à contenu identique | Apprendre la présence et non l'objet | C | Élevé | Élevé | P3, branche conditionnelle |
| E09 | Expert zones / expert trous | Séparer les deux régimes sans oracle | A | Moyen | Élevé | P2 |
| E10 | Prototypes locaux | Distances aux features propres/marquées | A | Faible | Moyen | P2 |
| E11 | Loss Jaccard symétrique | Mieux aligner apprentissage et score | A | Faible | Moyen | P4, déjà apparenté |
| E12 | Affinités spatiales prudentes | Corriger le bruit sans remplir tous les trous | A | Faible | Élevé | P3 |
| E13 | Cohérence des translations | Quantifier stabilité et artefacts de phase | B | Moyen–élevé | Moyen | P2 |
| E14 | Gradients contrastés | Sensibilité au vrai message versus contrôles | B | Moyen | Élevé | P3 |
| E15 | Ré-embedding contrefactuel | Comparer résidu observé et résidu attendu | B | Élevé | Très élevé | P4 |
| E16 | Réponse au retrait TrustMark | Une seconde vue du signal | C | Moyen | Élevé | P3 |
| E17 | Prior forensique externe | Utiliser les incohérences d'édition comme indice | C | Élevé | Élevé | P4 |
| E18 | Fusion complémentaire / distillation | Combiner des erreurs réellement différentes | A puis B/C | Moyen–élevé | Élevé | Après confirmations |

### E01 — Préserver le signe

**CODE** : `log1p(clamp(features,min=0))` efface toutes les valeurs négatives. **Hypothèse** : certaines sorties en contiennent et leur signe aide la séparation. Mesurer la proportion négative par carte sur le train, puis comparer à `sign(x)*log1p(abs(x))`, sans changer les dimensions. Contrôle identité et carte par carte. Abandon si aucune valeur négative ou aucun gain confirmé. Nouveauté limitée : diagnostic de représentation, pas nouvel algorithme.

### E02 — Pyramide et contrôle des raccourcis

Projections aux résolutions disponibles, fusion progressive 32→64→128 et ajout optionnel du RGB. Comparer avec/sans RGB à architecture constante, puis avec une tête de capacité comparable. Un gain sur contours sans gain sur l'intérieur ne suffit pas. Aucune interpolation ne crée de détail absent du cache. Idée de fusion inspirée de S5, architecture proposée spécifiquement ici.

### E03 — Signature locale du message

Pour des logits locaux `l_k` et signes `s_k=2m_k-1`, conserver `s_k*l_k`, leur dispersion et leur confiance, plutôt que deux moyennes. Ajouter une référence de messages aléatoires valides et un score d'écart normalisé. Tester vrai message versus message permuté, à poids fixes. Ne pas imposer le masque tout-zéro au mauvais message : la cible officielle est la présence du watermark, pas l'identité de son message. S1 motive la compatibilité, pas la validité d'une p-value locale.

### E04 — Géométrie StegaStamp

S3 révèle un point à vérifier : la transformation spatiale précède les couches choisies. Inspecter le graphe ONNX et ses coordonnées ; si nécessaire, remettre cartes ou logits dans le repère d'entrée avec une carte de validité. Contrôle transformation identité et motifs de coordonnées. Ne pas désactiver le STN pour « corriger » le modèle sans expérience séparée. Abandon si export sans transformation ou géométrie déjà compensée.

### E05 — Résidus multibandes

Calculer sur RGB 512 des filtres fixes de plusieurs échelles et orientations avant toute réduction, puis projeter ces indices dans une petite branche. Comparer aux seuls gradients RGB et à l'entrée résiduelle mélangée entre images. Choisir les échelles sur les données d'entraînement internes. Risque : apprendre textures, JPEG ou génération, pas watermark. S1/S2 ne justifient pas que tout le signal est haute fréquence ; S5 motive une seconde modalité seulement.

### E06 — Occlusion qui distingue signal et perturbation

Étendre l'occlusion existante en enregistrant les logits de chaque intervention, réutilisables avec plusieurs messages de contrôle. Mesurer le différentiel de compatibilité vrai/faux message ; comparer flou et moyenne locale à budget d'appels identique. Garder intensité brute et variation spatiale. Contrôles positions aléatoires et score indépendant du message. Les effets de normalisation et la redondance du message peuvent invalider le mécanisme. S6/S7 motivent le protocole de contrôle.

### E07 — Adaptation légère

Dégeler seulement le dernier bloc torch, ou ajouter de petits adaptateurs, sur les images des folds d'entraînement. Conserver une tête de masque et éventuellement un décodage local auxiliaire supervisé uniquement là où la cible contient du watermark. Au départ : segmentation seule, même tête et mêmes seeds. StegaStamp ONNX exclu du MVP faute de chemin différentiable vérifié. Ne jamais construire un cache commun après fine-tuning sur les 510 images avant la CV.

### E08 — Paires synthétiques à contenu identique

Sur une source propre autorisée, fabriquer une version marquée avec le bon embedder, puis les combiner sous des masques variés. Préentraîner la localisation avant les vraies images. Si seules des régions propres du train sont disponibles, préserver leur provenance par fold et ne pas supposer qu'un simple débruitage produit une source propre. Contrôle coutures seules et évaluation uniquement sur les vraies images OOF. S4 inspire la construction ; le domaine des traces du challenge peut différer fortement.

### E09 — Deux experts sans oracle

Un gate apprend sur le train à pondérer un expert « petites zones positives » et un expert « petits trous ». Entrées du gate : scalaires et features, jamais `full_watermark` au test. Comparer à la tête unique et aux deux seuils actuels ; surveiller les mauvaises décisions de régime. Déployer un mélange doux de probabilités. Ne pas nommer la couverture bimodale comme fait acquis sans audit des masques.

### E10 — Prototypes propres/marqués

Sur chaque train-fold, estimer plusieurs prototypes de features par classe avec échantillonnage équilibré par image. Les distances deviennent des canaux de la tête ou une baseline simple. Aucun pixel de validation dans les prototypes, PCA ou statistiques. Contrôle prototypes sans message ; mesurer le contraste local au-delà du score global. Hypothèse de peu de modes intra-classe possiblement fausse ; ni voisins ni prototypes ne doivent identifier les images sources.

### E11 — Loss ciblée

Essai distinct uniquement si l'ancien code e17 ne traitait pas deux classes par image : moyenne de Lovász hinge sur `(z,y)` et `(-z,1-y)` avec BCE. Vérifier les masques vides/pleins et la stabilité des gradients. Le journal fournit déjà un résultat défavorable pour Lovász+BCE ; ne pas relancer sans différence scientifique explicite. Priorité derrière les nouvelles entrées.

### E12 — Affinités, pas nettoyage aveugle

Propager les probabilités entre cellules voisines ayant des features proches, avec force limitée et identité comme contrôle. Les bords RGB servent au plus de garde-fou. Mesurer particulièrement les petites composantes et trous : leur suppression peut effacer la vérité. Le post-traitement par aire a déjà été peu convaincant ; seule une relation de voisinage différente justifie l'essai.

### E13 — Stabilité sous translations

Conserver plusieurs vues de features, réalignées dans les coordonnées originales, puis leur moyenne et variance. Comparer décalages circulaires et padding réfléchi avec zones valides. Ne pas faire de TTA en retournant uniquement les caches : le modèle peut ne pas être équivariant. Un champ réceptif large empêche une garantie simple sur la largeur à exclure près des bords. Coût limité par un petit nombre de vues préfixé.

### E14 — Gradients contrastés

Comparer gradients de compatibilité vrai message, faux messages et poids randomisés, au lieu d'une seule valeur absolue. Tester une contribution signée sur chemin image→image floutée ; distinguer sensibilité et présence. Utiliser uniquement un chemin différentiable réel ; les wrappers `no_grad` doivent être contournés dans un adaptateur isolé et vérifié. S7 fournit les contrôles, pas une preuve que cette méthode localise.

### E15 — Ré-embedding contrefactuel

Calculer le résidu ajouté en ré-encodant l'image avec le message connu puis avec des messages de contrôle ; corréler localement avec des résidus d'image. Hypothèse risquée de stabilité du motif sous ré-embedding. Le réseau dépend du contenu, de l'échelle et d'un watermark déjà présent. Contrôle même image, message faux et gain de simple énergie résiduelle. Inspection préalable de l'API embed ; aucune signature inventée.

### E16 — Retrait comme seconde mesure

Pour TrustMark seulement, extraire la différence avant/après réseau de retrait compatible et la variation d'accord. Superviser leur utilité avec les vraies cibles, sans traiter cette différence comme vérité. Catégorie C tant que poids et autorisation non confirmés, B s'ils font partie du starter autorisé. Contrôles débruitage générique et zones propres. S2 motive l'opérateur, pas sa précision comme localisateur.

### E17 — Prior forensique externe

Tester une carte figée TruFor comme canal auxiliaire, jamais comme masque final ni son complément systématique. Évaluer si elle aide sur les images où l'accord watermark est faible. Catégorie C : modèles externes et licence à vérifier. Abandon si le gain vient seulement de contours sémantiques ou si coût/distribution ne conviennent pas. Le pré-entraînement sur traces de caméra peut être peu adapté à des images générées.

### E18 — Fusion puis distillation

Mesurer les deltas mIoU par image et la corrélation d'erreurs de branches confirmées. Tester d'abord moyenne fixe, sans sélection par masque. Un méta-modèle nécessite un cross-fitting imbriqué : les features OOF du train externe ne doivent pas provenir de modèles entraînés sur les labels de la validation externe. La distillation utilise un teacher propre au fold, jamais un teacher ayant vu ses cibles de validation. A si uniquement calculée à la volée dans train.py avec caches actuels ; B/C selon les nouveaux signaux requis.

## 4. Protocole commun et limites statistiques

1. **Geler la référence** : commit et hachages ci-dessus, folds existants, ordre des noms, caches et sidecars explicitement manifestés. Aucun chargement facultatif silencieux pendant une comparaison. Un changement de harness reçoit son propre ID.
2. **Apparier** : candidat et référence mêmes images, seeds, epochs et seuils. Garder le comportement deux seuils pour la comparaison historique, plus un score à seuil 0,5 sans sélection. Une référence un seuil constitue une ablation séparée.
3. **Explorer économiquement** : un seed pour filtrage, trois folds conservés ; comparaison avec référence un seed. Confirmer ensuite avec les trois seeds de référence. Ne pas comparer un candidat un seed à l'ensemble historique.
4. **Conserver les sorties** : probabilités OOF alignées, fold, seed, seuil(s), par-image `IoU0`, `IoU1`, mIoU, couverture prédite/réelle, temps extraction/train/inférence, mémoire torch ET consommation ONNX si utilisée. Le compteur torch n'inclut pas automatiquement toute la mémoire ONNX.
5. **Ne pas sélectionner sur les folds évalués pour une estimation confirmatoire** : epochs fixées, transformations apprises uniquement dans le train-fold. Pour calibrer proprement les seuils d'un fold externe, produire des prédictions internes sur son train seulement, puis appliquer ces seuils au fold externe intact. Les simples seuils choisis sur l'OOF restent exploratoires.
6. **Petites régions** : score par méthode, couverture `[0,.05)`, `[.05,.5)`, `[.5,.95)`, `[.95,1]`, cas vide/plein, composantes et largeur des régions. Afficher effectifs ; les labels servent ici à l'analyse, pas à choisir la sortie d'une image.
7. **Confirmer les gagnants** : différences appariées par image, moyenne et médiane, victoires/défaites, bootstrap par image stratifié par méthode (par source si groupes connus), intervalle 95 %, seeds supplémentaires pour sensibilité. Les cellules corrélées ne sont pas des observations indépendantes. L'intervalle conditionnel ne corrige ni la sélection de nombreuses branches ni une dépendance inconnue entre images.
8. **Fuites** : variantes d'une source dans le même groupe si identifiables ; synthétiques confinés au train-fold ; aucun ajustement sur les images test. Si nouveau split de confirmation, figer sa définition et rerun référence+candidat, sans remplacer les scores historiques. Un nouveau split des mêmes 510 images n'est pas un jeu réellement indépendant après exploration répétée.
9. **Rejet ou conservation** : conserver une piste pour confirmation si l'amélioration appariée dépasse sa variabilité observée ou si elle apporte une complémentarité mesurée. Pas de règle universelle « +0,0001 = progrès ». Documenter tout compromis par méthode et coût. Réussite technique et réussite scientifique sont deux statuts distincts.

L'autoresearch écrit actuellement « garder si strictement meilleur ». Ce protocole de recherche ne change pas cette boucle ; il ajoute une confirmation avant de conclure scientifiquement. Pas de garantie non biaisée avec seulement ces données déjà explorées.

## 5. Ordre recommandé

**Trois premières branches : E01, E03, E04.** E01 teste à faible coût une perte potentielle d'information ; E03 exploite le message connu plus complètement ; E04 vérifie une hypothèse géométrique avant de complexifier StegaStamp. L'inspection de E04 peut conclure rapidement qu'aucune correction n'est nécessaire.

Si les features sont non négatives, E01 devient un contrôle et E02 prend sa place. Si les logits locaux ne sont pas réellement exposés, différer E03 et passer à E05. Si StegaStamp est déjà aligné, remplacer E04 par E05/E06 selon le coût de décodage mesuré. Si l'échec reste dominé par les petites régions, privilégier nouvelles cartes fines plutôt que cinq seeds supplémentaires.

Informations manquantes : version/hash des wrappers et checkpoints ; tenseurs et ranges réels ; détail ONNX/STN ; présence et provenance des sidecars ; logs bruts et OOF ; quasi-doublons ; règles des données synthétiques et modèles externes ; budget GPU encore disponible. Elles bloquent certaines implémentations conditionnelles, pas le catalogue.

Voir [les huit spécifications](BRANCH_SPECS.md), puis [les prompts pour Claude Code](CLAUDE_HANDOFF.md). Les documents proposent des branches ; aucune n'a été créée ou exécutée.
