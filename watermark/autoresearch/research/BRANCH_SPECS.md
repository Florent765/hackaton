# Huit branches expérimentales à implémenter

Spécifications, pas résultats. Référence : `26e5e0db4b6b34139c6d2396958a26fa22e280d5`, audit du 21/11/2026. Voir [l'audit, les sources et les 18 pistes](SOLUTION_LANDSCAPE.md). Les numéros E renvoient à ce catalogue.

## Contrat commun

Chaque branche part de la même référence dans un worktree indépendant. Ne pas changer de branche dans un checkout partagé ou faire de reset destructeur. Lire les instructions locales ; préserver les changements existants. Le travail de cette revue n'a créé aucun worktree ni branche.

**Invariants** : folds existants, métrique officielle, noms et ordre, entrée 512, sortie 128, trois méthodes connues, train/test disjoints. Les types numériques/ranges des wrappers sont à inspecter. Les messages complets ne sont pas dans le cache principal. Aucune statistique apprise sur le fold évalué ni sur le test. `full_watermark` n'est pas une entrée d'inférence.

**Interfaces CODE disponibles** : `prepare.load_cache(method, split)`, `prepare.evaluate_oof(preds)`, `prepare.write_submission(names,masks,path)`, `baseline.FrozenExtractor`, `baseline.extract_all`. Les interfaces `wmloc.data.LocalisationDataset` et `wmloc.watermarks.load_watermark` sont appelées par le code local mais leur implémentation est à vérifier sur le starter. Ne pas inventer `decode_features`, `embed` ou une conversion ONNX différentiable.

**Disponibilité** : les scripts utilisent torch/numpy, ONNX et ONNX Runtime pour StegaStamp. Cela ne garantit pas leur présence ou version dans l'environnement d'exécution. A : aucune installation. B : réutiliser les paquets existants, annoncer tout manque sans modifier la `.venv` partagée. C : vérifier autorisation, disponibilité et licence des nouveaux assets.

**Isolation** : en B/C, créer `autoresearch/experiments/E##/` et un cache distinct `ar_cache_research/E##/<manifest_hash>/`. Ne pas modifier `baseline.py`, `prepare.py`, `worker.py` ni leurs caches partagés. Le worker fixe importe son harness par PYTHONPATH : ne pas supposer qu'un fichier local homonyme surcharge correctement `prepare`. Un entry point isolé importe explicitement le harness fixe pour folds/évaluation, et ses propres modules sous des noms distincts.

**Manifestes** : hash du code d'extraction, wrappers, checkpoints et métadonnées ; formes/dtypes ; normalisation ; taille et interpolation ; repère géométrique ; méthode ; IDs/noms des images ; seeds ; présence obligatoire ou absence déclarée des sidecars. Un cache manquant ne doit pas modifier silencieusement l'architecture. Ne jamais utiliser les fichiers d'un job encore actif comme destination d'une mise à jour.

**Validation minimale** : un petit batch réel par méthode concernée, shapes/finitude, gradients tête et gel extracteur, comparaison métrique officielle, entraînement sur un mini-ensemble pour détecter une incapacité à apprendre. Aucun gain scientifique déduit de ce smoke test. Rejouer candidat et référence sous mêmes folds/seeds/seuils. Sauver probabilités OOF, scores individuels et coûts. Voir le protocole complet du paysage pour la calibration interne et le biais de sélection.

**GPU** : un job maximum pour toute l'équipe, en cours ou en attente. Vérifier l'état d'équipe, pas seulement son utilisateur ; respecter la contrainte la plus stricte si les anciennes recettes d'enchaînement sont contradictoires. Pas de nouveau job tant qu'un job existe. Réutiliser l'allocation autorisée ou attendre sa fin. Benchmark extraction séparé du budget d'essai ; aucun calcul lourd sur login. Commencer avec un worker par GPU pour des mesures de coût interprétables.

## E01 — `research/wm-e01-signed-features` — A

**Hypothèse** : l'écrêtage des activations négatives détruit un signal discriminant dans certaines cartes.

**Motivation** : observation directe de `train.inputs()`, pas revendication issue d'un papier. Le journal montre un gain de log1p, mais ne distingue pas compression et suppression du signe.

**Pipeline** : cache brut → transformation signée → tête actuelle → probabilités → seuils actuels.

**Entrées** : `feats[name]` `[N,C,H,W]` pour `feature_names`, tailles C/H/W lues du cache. RGB `[N,3,64,64]`, scalaires `[N,2]`, accord et occlusion inchangés. Les float16 originaux peuvent contenir des négatifs ; ne pas l'affirmer avant mesure.

**Implémentation** : seul `train.py` ; ajouter une option de transformation `current|signed|identity`. `signed(x)=sign(x)*log1p(abs(x))`. Instrumenter taux de négatifs et quantiles par carte à partir du train, sans prétendre que ces statistiques prouvent un signal. Pas de standardisation supplémentaire au premier essai.

**Apprentissage/message** : tête, BCE+Dice, epochs/seeds inchangés ; l'usage du message reste celui de la référence. Tester globalement puis carte/méthode responsable, sans recherche exhaustive.

**Contrôle décisif** : pour x≥0, résultat identique à `current` ; pour x<0, signe conservé. Comparer même nombre de paramètres et même seuil fixe 0,5, puis seuils exploratoires identiques.

**Succès/réfutation** : gain apparié reproductible sur les méthodes contenant des négatifs. Si aucune activation négative, les variantes doivent coïncider et la branche s'arrête. Si seuls les seuils changent, pas de preuve d'amélioration de la représentation.

**Coût** faible ; risque faible ; aucune dépendance. Variante n'ajoutant ni cache ni paquet.

## E02 — `research/wm-e02-native-pyramid` — A

**Hypothèse** : une fusion progressive des cartes conserve mieux les petits détails qu'une concaténation unique à 64, et limite les raccourcis RGB.

**Motivation** : fusion multimodale de [TruFor, §3.2](https://arxiv.org/html/2212.10957v2#S3.SS2), transférée ici sous une forme légère conçue pour nos caches ; pas de promesse de transférer ses résultats.

**Pipeline** : projections par carte → fusion successive 32,64,128 → logits. À chaque niveau, ajouter seulement les sources dont la résolution est adaptée ; ne pas appeler une carte interpolée « nouvelle feature haute résolution ».

**Entrées** : cartes principales et accord existants ; RGB et scalaires identiques. Ordre NCHW à valider, surtout StegaStamp. Dimensions réelles lues ; ne pas hardcoder les valeurs de canaux.

**Implémentation** : `train.py` uniquement ; fusion additive ou concaténation compacte, width initiale 32, projections 16. Redimensionner explicitement RGB à chaque branche. Conserver le nombre de paramètres et la mémoire mesurés pour une ablation de capacité.

**MVP** : pyramide sans nouvelle attention, BCE+Dice inchangée. Puis ablation `RGB off` versus `RGB on`, sur la même architecture. Aucun changement de loss ou de seuil au même moment. Message : cartes d'accord inchangées.

**Contrôles** : tête actuelle, pyramide, pyramide sans RGB ; capacité comparable dans une confirmation si nécessaire. Mesurer IoU des zones fines et des trous, pas uniquement contours visuels. Si le RGB seul explique le gain, documenter ce raccourci potentiel.

**Succès/réfutation** : amélioration des petites régions confirmée sans régression majeure ailleurs. Rejeter si coût augmente sans meilleur score ni complémentarité. Les caches grossiers peuvent être la vraie limite.

**Coût** faible–moyen ; risque moyen ; aucune dépendance scientifique à E01 (combinaison ultérieure seulement).

## E03 — `research/wm-e03-message-evidence` — B

**Hypothèse** : la distribution des accords par bit permet de distinguer un signal cohérent d'une moyenne élevée produite par quelques logits dominants.

**Motivation** : [Pixel Seal §3.1](https://arxiv.org/html/2512.16874v1#S3.SS1) et [WAM §3](https://arxiv.org/html/2411.07231v2#S3) motivent l'utilisation du message ; les statistiques ci-dessous sont nos propositions.

**Pipeline** : image + message → wrapper figé → logits locaux complets → statistiques de preuve → concaténation aux entrées habituelles → tête.

**Entrées nouvelles** : `local_logits [N,K,h,w]`, `message [N,K]`, mêmes `names`. K/h/w mesurés (256 et 100 bits annoncés respectivement pour PixelSeal/TrustMark, à confirmer). StegaStamp inchangé : ne pas inventer des logits locaux après ses couches denses. Conserver d'abord le shift-and-stitch de référence, pour ne pas mélanger deux modifications.

**Fichiers** : `experiments/E03/prepare_evidence.py`, `train_evidence.py`, manifeste ; import du harness fixe. Aucun écrasement des cartes v1/v2.

**MVP** : calculer `r_k=(2m_k-1)*l_k`, moyenne, dispersion, quantile bas, fraction positive et `Z=sum(s*l)/sqrt(sum(l²)+eps)`. Ajouter quelques canaux scalaires spatiaux, sans grande tête. Comparer au seul accord moyen avec même capacité.

**Contrôles mathématiques** : sous signes aléatoires indépendants, le dénominateur de Z est la dispersion conditionnelle du produit scalaire ; ce n'est PAS une distribution nulle garantie sur images propres. Des bits ECC/corrélés invalident l'interprétation naïve. Comparer des messages réellement valides si leur codage est connu. Une simple différence entre log-vraisemblance vrai message et messages symétriques peut se réduire à l'accord linéaire déjà présent : démontrer qu'un canal apporte autre chose avant de le retenir.

**Deuxième étape conditionnelle** : projection apprise des canaux r_k (1×1), régularisée ; tester identité des bits permutée de façon cohérente versus simple moyenne. Aucun objectif « mauvais message = masque zéro » ; utiliser les mauvais messages comme contrôle d'évidence.

**Apprentissage** BCE+Dice, mêmes folds. Normalisations apprises sur train-fold uniquement. Contrôles vrai/faux message à poids fixes et canal de variance supprimé.

**Succès/réfutation** : gain apparié et dépendance au vrai message. Si faux messages font aussi bien, le mécanisme revendiqué n'est pas établi. Coût moyen extraction/cache, entraînement faible–moyen ; risque moyen ; bloqué si logits locaux non accessibles ou sémantique non vérifiée.

## E04 — `research/wm-e04-stega-coordinates` — B, conditionnelle

**Hypothèse** : des features post-transformation spatiale sont comparées à un masque dans un autre repère, réduisant la localisation.

**Source** : [StegaStamp §4.2](https://conferences.computer.org/cvpr/pdfs/CVPR2020-1XMljIyuXWg2zC9bnL19Tw/716800c114/716800c114.pdf) et [StegaStampDecoder.call](https://raw.githubusercontent.com/tancik/StegaStamp/master/models.py). L'export du challenge peut différer.

**Étape préalable obligatoire** : inspecter l'ONNX réel. Déterminer présence du STN, entrée/sortie du sampler, paramètres, repère et ordre des axes. Mesurer si la transformation est identité ; comparer les logits avec ceux du wrapper. Si absente/déjà compensée, rendre un constat négatif et ne pas entraîner.

**Pipeline proposé** : image → STN → features → reprojection dans le repère original + validité → tête de référence. Ne pas remplacer le STN par l'identité, ce serait une autre méthode.

**Fichiers** : `experiments/E04/inspect_geometry.py`, `prepare_geometry.py`, `train_geometry.py`; copie ONNX avec sorties auxiliaires, cache distinct. Ne pas supposer le nom d'un tenseur theta absent du code local.

**Tenseurs** : features NCHW une fois axes validés ; paramètres `[N,2,3]` seulement si le graphe est effectivement affine ; grilles spatiales sinon. Géométrie exprimée en coordonnées normalisées ou pixels selon sampler réel. `grid_sample` sert de piste d'implémentation, pas de preuve de même convention que TensorFlow.

**MVP/tests** : images de coordonnées, impulsion, translation connue, identité et aller-retour sur domaine valide. Si le sampler donne `p_input=T(p_rectified)`, construire la grille inverse pour remonter vers l'entrée et vérifier numériquement, sans inversion intuitive. Masquer les points hors champ ; n'en faire ni zéro propre ni vérité terrain. Ajouter le masque de validité comme entrée.

**Contrôle** : même feature, même tête, seule reprojection change. BCE+Dice inchangée ; message via scalaires et sidecar existants. Comparer erreur versus amplitude de déformation.

**Succès/réfutation** : correction de géométrie vérifiée avant mIoU. Transformer presque identité ou aucun gain = abandon de l'hypothèse. Coût moyen ; risque de nouvelle erreur de repère ; dépend uniquement de l'export réel.

## E05 — `research/wm-e05-native-residuals` — B

**Hypothèse** : des traces utiles sont détruites par le pooling RGB à 64, et une branche résiduelle légère peut les récupérer.

**Motivation** : [TruFor §§3.1–3.2](https://arxiv.org/html/2212.10957v2) pour la séparation des modalités ; filtres fixes proposés ici, pas réimplémentation de Noiseprint++.

**Pipeline** : RGB float32 512 → différences gaussiennes et dérivées orientées → regroupement local → cache 128 → branche 16 canaux → fusion tardive avec tête → masque.

**MVP proposé** : 3 échelles de résidu `x-G_sigma(x)`, sigma 1,2,4 pixels d'entrée, et dérivées horizontale/verticale. Garder moyenne signée et RMS local sur blocs 4×4, avec nombre exact de canaux documenté. Toute autre échelle est une expérience ultérieure. Ne pas extraire les résidus depuis RGB64.

**Fichiers** : `experiments/E05/prepare_residuals.py`, `train_residuals.py`. PyTorch uniquement si présent ; aucune dépendance à TruFor ni poids externe. Messages et caches de base inchangés. Normalisation robuste estimée sur le train-fold seulement, conservée au test.

**Objectif** : BCE+Dice de référence ; nouvelle branche compacte, garder comparaison capacité seule. PixelSeal, TrustMark, StegaStamp entraînés séparément ; mêmes filtres initiaux pour limiter les choix.

**Contrôles** : baseline ; baseline+résidus ; baseline+gradients RGB seuls ; résidus mélangés entre images de même méthode comme contrôle diagnostic. Analyser contenu texturé versus uniforme et couverture. Les résidus ne sont pas une vérité terrain ; ne pas les seuiller comme masque officiel.

**Succès/réfutation** : amélioration réelle sur nouvelles images, non expliquée seulement par contours/texture. Coût moyen extraction, faible–moyen tête ; risque moyen de raccourcis ; pas de dépendance à E03.

## E06 — `research/wm-e06-counterfactual-probes` — B

**Hypothèse** : l'effet d'une intervention sur l'accord au vrai message, corrigé par des messages de contrôle, fournit un signal plus spécifique que del8 seul.

**Références** : [RISE](https://cse.iitkgp.ac.in/~adas/papers/BMVC_2018_RISE.pdf), [tests de saillance](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf). Adaptation proposée, pas reproduction de RISE.

**MVP** : mêmes 64 positions de del8, conserver tous les logits `L [N,65,K]` (original + interventions) et positions. Score brut `a(l,m)=mean((2m-1)*l)` ; garder aussi le score normalisé actuel comme contrôle. Delta vrai message, dispersion des deltas sous messages contrôles et delta standardisé deviennent des cartes 8×8. Les nouveaux messages ne nécessitent pas de nouveau passage du modèle pour ces logits déjà calculés.

**Deuxième facteur séparé** : flou versus moyenne locale, jamais deux opérateurs et grille plus fine simultanément. Éviter faux labels « flou=propre ». Une carte moyenne sur mauvais messages peut être mathématiquement nulle : calculer variance/contrastes utiles plutôt qu'ajouter un canal redondant.

**Fichiers** : `experiments/E06/prepare_probes.py`, `train_probes.py` et cache propre. Réutiliser la logique de `prepare_occ.py` sans la modifier. Alignement exact des noms ; s'assurer d'un provider GPU ONNX réel. PixelSeal/StegaStamp d'abord ; TrustMark seulement après probe informatif, contrairement à une généralisation automatique.

**Apprentissage** : concaténation des nouvelles cartes à tête constante, BCE+Dice. Comparer au del8 brut et normalisé (e33 déjà préparé) avec budget d'interventions égal. Les messages de contrôle servent à l'analyse du signal, pas à changer le label officiel.

**Tests** : insertion/suppression sans déplacement, carte constante, message faux, régions aléatoires, AUROC avec ex æquo bien gérés ; la sélection finale reste la mIoU. Contrôler que les effets ne suivent pas uniquement l'énergie du blur.

**Succès/réfutation** : nouveau score plus discriminant ET gain OOF/coût acceptable. Échec si redondance globale rend les variations nulles ou si l'opérateur domine. Coût élevé en extraction, faible après cache ; aucun agrandissement automatique de grille.

## E07 — `research/wm-e07-local-adaptation` — B

**Hypothèse** : la supervision spatiale peut adapter légèrement les représentations globales là où une tête seule plafonne.

**Motivation** : [WAM §4.2](https://arxiv.org/html/2411.07231v2#S4.SS2) inspire une supervision locale, sans supposer que ses poids transfèrent.

**Pipeline** : image native → prétraitement officiel → extracteur torch avec dernier bloc entraînable → tête → logits 128. Versions congelée et adaptée exactement appariées. PixelSeal/TrustMark seulement au départ ; StegaStamp ONNX exclu.

**Fichiers** : `experiments/E07/train_adaptation.py`, adaptateur différentiable isolé ; pas de modification du wrapper partagé. Vérifier le chemin brut correspondant à `features/logits` et la parité numérique avant gradient. Ne pas passer par `FrozenExtractor.__call__` décoré no_grad.

**MVP** : segmentation seule, dernier bloc dégelé, LR extracteur 1/10 du LR tête comme configuration de départ, BatchNorm en eval, reste gelé. Garder le même nombre de vues que la référence pour une comparaison stricte ; si réduction pour coût, comparer à une référence congelée utilisant la même réduction.

**Extension séparée** : loss auxiliaire de bits sur régions marquées, avec cible spatiale pondérée selon recouvrement, sans affecter aux cellules grossières une certitude que leur champ réceptif ne justifie pas. Sur régions propres, ne pas apprendre un message arbitraire ni copier le message fourni via une voie triviale.

**Tests** : parité wrapper, gradients uniquement autorisés, labels/coordonnées, aucun poids partagé entre folds. Chaque fold repart du checkpoint officiel. Caches de features apprises propres à ce fold seulement ; aucun cache global issu d'un entraînement sur tout le train.

**Succès/réfutation** : gain sur validation externe appariée, pas seulement train ; sinon revenir aux poids figés. Coût élevé, budget séparé du worker 600 s si nécessaire ; risque fort de surapprentissage et de modification du signal. Aucun nouveau modèle préentraîné requis si interfaces utilisables.

## E08 — `research/wm-e08-matched-synthesis` — C, conditionnelle

**Hypothèse** : des paires de contenu identique avec et sans signal enseignent une discrimination plus pertinente que davantage d'epochs sur 170 images.

**Motivation** : [WAM §4.2 et annexe D.2](https://arxiv.org/html/2411.07231v2), transfert du principe de composition vers les embedders du challenge. Cette synthèse n'a pas été validée sur notre distribution.

**Prérequis** : règles autorisant synthèse et éventuelles sources externes ; versions exactes des embedders ; sources propres dont la provenance est établie. Les images test ne sont jamais des sources. Pas d'import massif de données ni réentraînement de l'embedder avant cette vérification.

**Construction** : source propre x et message m → image watermarkée w via API vérifiée → `x_mix=M*w+(1-M)*x`. Le masque M est généré à 512 puis réduit par la fonction officielle. Chaque fold synthétise uniquement depuis ses propres sources train, variantes regroupées avec leur origine. Priorité aux masques irréguliers, trous et petites régions, sans recopier la distribution mesurée du fold évalué.

**Fichiers** : `experiments/E08/generate_matched.py`, `train_synthetic.py`, registre de provenance ; nouveau dataset/cache. Tête de référence pour le MVP, préentraînement synthétique puis entraînement réel. Garder équivalent le nombre d'étapes avec une baseline « réel répété », afin de séparer effet données et calcul.

**Message** : changer m entre synthèses d'une même source ; employer le format réellement attendu par l'embedder, y compris ECC éventuel. Aucun appel `embed` supposé. Interpolation et force alignées avec l'organisateur ; décalage inconnu signalé.

**Contrôles** : mêmes coutures sur compositions propre/propre ; annotations nulles pour ce contrôle propre ; mesure sur vraies images OOF uniquement. Ne pas considérer des photos floutées ou passées par un remover comme parfaitement propres sans preuve. Si seules des zones propres découpées sont accessibles, expliciter le changement de distribution.

**Succès/réfutation** : gain sur vraies images et plusieurs seeds ; échec si gain synthétique seulement ou dépendance aux coutures. Coût élevé, risque élevé de fuite et domaine ; branche non démarrable tant que règles/sources manquent. Peut ensuite alimenter E07, mais premier test indépendant.

## Matrice de lancement

| Première vague | Condition de passage | Suite |
| --- | --- | --- |
| E01 | Valeurs négatives présentes | Comparaison contrôlée, sinon E02 |
| E03 | Logits locaux et messages bruts accessibles | MVP statistiques puis projection par bit |
| E04 | STN exporté, coordonnées non déjà compensées | Reprojection ; sinon rapport négatif puis E05 |
| E05 / E06 | Budget d'extraction mesuré acceptable | Comparaison séparée puis fusion éventuelle |
| E07 | Chemin différentiable vérifié, budget disponible | Adaptation segmentation seule |
| E08 | Règles + provenance validées | Synthèse pilote puis évaluation réelle |

Toutes les branches conservent des observations négatives utiles. Aucune stratégie ne doit être qualifiée de nouvelle au sens de la littérature : il s'agit d'adaptations proposées pour ce pipeline.
