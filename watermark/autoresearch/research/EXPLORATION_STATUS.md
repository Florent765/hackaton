# Registre d'exploration E01–E18

Mis à jour en continu. Référence commune : **REF** = `train.py` @ `26e5e0d` (e29) rejoué par `experiments/common/rlib.py`
(différences explicites : sidecars déclarés `occ` pour PixelSeal/StegaStamp, aucun chargement implicite ; un seed
par run ; calibration interne au train-fold). Empreintes vérifiées : train.py `56f0015f…`, prepare.py `017f7088…`,
baseline.py `e85a540c…` ; JURECA : wrappers `faeff2ca…`, PixelSeal ckpt `0c5665cf…`, StegaStamp ONNX `e7ade660…`,
TrustMark Q decoder `e3d9cea5…`. Caches v1 + sidecars copiés sur `/p/project1/training2666/jacquet2/ar_cache`
(scratch du projet saturé en inodes depuis ~20h50).

Trois politiques de seuil : `fixed05` (0,5), `explor` (deux seuils choisis sur l'OOF : exploratoire), `confirm`
(deux seuils choisis par CV interne 2 plis sur le train-fold, appliqués au fold externe : confirmatoire).
Filtrage : 1 seed apparié avec REF 1 seed ; confirmation : 3 seeds appariés. Décision sur `confirm`.

**REF (seed 0)** : fixed05 0,5002 · explor 0,5018 · confirm 0,5006 · 61 s · 2,3 Go. (Le e29 historique 3 seeds,
explor, valait 0,5079 : l'écart vient du nombre de seeds.)

## Diagnostics transverses (D1)
- **Couverture** : régimes séparés à 0,5 (full_watermark=False : max 0,44 ; True : min 0,51), mais étalés à
  l'intérieur ; « bimodal » = deux régimes disjoints, pas deux pics à 0 et 1. Aucun masque vide ni plein.
- **AUROC par image, ex æquo moyennés (510 images)** : PixelSeal agree 0,602 / occ 0,582 / occ16 0,549 ;
  TrustMark agree 0,586 ; StegaStamp occ 0,516 (l'ancien probe sur 24 images, sans ex æquo, donnait 0,61).
- Messages : bits non corrélés (|r| moyen 0,061 vs 0,077 attendu au hasard) → pas de structure ECC visible.
- `analyze_oof.py` historique attend un seuil scalaire : non utilisé ; `rlib.summarize` gère les deux seuils.

## Tableau

| ID | Hypothèse | Cat. | Dépend. | Branche / commit | Expérience minimale | Statut | Résultats / preuves | Coût mesuré | Décision / suite |
|---|---|---|---|---|---|---|---|---|---|
| E01 | L'écrêtage `log1p(clamp(x,0))` détruit un signal utile | A | — | research/wm-e01-signed-features | `signed` vs REF, 1 seed apparié | **rejeté** | Négatifs : PixelSeal convnext 50 %, embedding 25 % (GELU, min −0,17) ; TrustMark/StegaStamp 0 % (contrôle : delta exactement 0). confirm Δ −0,0013 IC95 [−0,0046 ; +0,0015], W/L 167/135, PixelSeal −0,0036 ; explor Δ −0,0001 | 60 s, 2,3 Go | Le signe n'aide pas la tête. `identity` non lancé (même question, valeurs ±3000 non compressées : faible valeur). Clos. |
| E02 | Fusion pyramidale 32→64→128, contrôle RGB | A | — | research/wm-e02-native-pyramid | pyramide vs REF ; pyramide sans RGB | à examiner | e34 (grille 32) négatif dans results.tsv : différence explicite requise (fusion multi-échelle, pas grille) | — | Après E03/E04 |
| E03 | Statistiques par bit (std, q10, Z) ajoutent un signal spécifique au message | B | logits locaux (✔ D1) | research/wm-e03-message-evidence | REF + 3 canaux vrai message vs mêmes canaux message aléatoire | **en cours** | logits locaux 256×8×8 / 100×8×8, moyenne = logit global (écart 6e-6) | extraction en cours | — |
| E04 | Cartes StegaStamp post-STN désalignées avec le masque | B | STN exporté (✔) | research/wm-e04-stega-coordinates | reprojection inverse + validité vs REF | **en cours** | STN `gen_stn` présent, θ = biais identité + MatMul ; déplacement médian 21 px (p90 30, max 51) @400 ; convention θ : sortie→entrée (aller-retour corr 0,983 vs 0,57) | inspection < 1 min | Reprojection lancée dès θ extraits |
| E05 | Résidus multibandes natifs 512 récupèrent des traces | B | — | research/wm-e05-native-residuals | branche résidus + contrôles gradients RGB / permutation | à examiner | — | — | Après E03/E04 |
| E06 | Occlusion différentielle vrai/faux message plus spécifique que del8 | B | logits des interventions | research/wm-e06-counterfactual-probes | cartes delta vrai − dispersion contrôles | à examiner | occ seul : AUROC 0,58 (PS) / 0,52 (SS) | — | Priorité selon E03 (même logique de contrôle) |
| E07 | Dégeler le dernier bloc améliore les cartes | B | chemin différentiable (PS/TM ✔ via wrappers internes) | research/wm-e07-local-adaptation | dernier bloc + tête, fold-isolé | à examiner | — | — | Coût élevé ; après signaux B |
| E08 | Paires synthétiques à contenu identique | C | règles/sources | research/wm-e08-matched-synthesis | — | bloqué par un prérequis | Règles sur données synthétiques / embedders non documentées dans le PDF ; aucune source propre garantie | — | Demander autorisation organisateurs ; voir spec |
| E09 | Deux experts (petites zones / petits trous) + gate appris | A | — | research/wm-e09-experts | voir spec ci-dessous | à examiner | e29 (deux seuils par régime) capte déjà une partie | — | — |
| E10 | Distances à des prototypes propres/marqués | A | — | research/wm-e10-prototypes | voir spec | à examiner | — | — | — |
| E11 | Lovász symétrique deux classes | A | — | — | — | différé | e17 (Lovász + BCE, deux classes par image, cf. code e17_lovasz) négatif : même mécanisme | — | Pas de différence scientifique disponible → non relancé |
| E12 | Propagation par affinités de features | A | — | research/wm-e12-affinity | voir spec | à examiner | e12 (lissage) et e35 (composantes) négatifs ; mécanisme différent (affinités de features) | — | — |
| E13 | Cohérence sous translations (moyenne/variance de vues) | B | ré-extraction | research/wm-e13-shift-consistency | voir spec | à examiner | — | — | — |
| E14 | Gradients contrastés vrai/faux message | B | chemin différentiable | — | voir spec | à examiner | gradient absolu : AUROC ≈ 0,49–0,53 (probe) | — | Faible priorité |
| E15 | Ré-embedding contrefactuel | B | API embed (✔ wrappers) | — | voir spec | à examiner | — | — | Risque très élevé |
| E16 | Réponse au remover TrustMark | C→B | remover présent dans le paquet trustmark (poids `trustmark_rm_Q` téléchargés) | research/wm-e16-remover | voir spec | à examiner | Poids du remover présents dans la copie privée : catégorie B (même paquet que l'extracteur) | — | — |
| E17 | Prior forensique externe (TruFor) | C | poids externes, licence | — | — | bloqué par un prérequis | Poids externes non présents, nœuds sans internet, règles non vérifiées | — | Autorisation + téléchargement requis |
| E18 | Fusion / distillation de candidats complémentaires | A→B | ≥2 candidats confirmés | — | moyenne fixe des probas OOF | bloqué par un prérequis | attend des candidats confirmés | — | — |

## Spécifications courtes (pistes sans fiche Codex détaillée)

- **E09 — experts + gate.** H : deux têtes spécialisées (images à zone positive minoritaire / à trous) mélangées par
  un gate appris sur les scalaires (bit accuracy, accord) surpassent la tête unique. Réfutation : Δconfirm ≤ 0 ou
  gate ≈ seuil bit accuracy déjà utilisé. MVP : 2 têtes REF entraînées sur tout le train-fold avec pondération de la
  loss par régime prédit (pas full_watermark), mélange doux. Coût ≈ 2× REF.
- **E10 — prototypes.** H : distance (cosinus) des features projetées à k=4 prototypes par classe estimés sur le
  train-fold seul ajoute un contraste local. Réfutation : AUROC canal ≤ agree ou Δconfirm ≤ 0. MVP : k-means sur
  features log1p sous-échantillonnées (équilibré par image), 2 canaux de distance. Coût faible.
- **E11 — Lovász symétrique.** Différé : e17 testait déjà Lovász hinge/softmax deux classes par image + BCE (Δ −0,003).
- **E12 — affinités.** H : propagation des probabilités entre cellules voisines aux features similaires (1–2
  itérations, force α choisie en CV interne) corrige du bruit sans combler les vrais trous. Réfutation : Δconfirm ≤ 0
  ou baisse sur couverture < 5 %. Coût : post-traitement GPU léger.
- **E13 — translations.** H : la variance entre les 16 vues du shift-and-stitch indique les zones de watermark
  (réponse stable) vs propres (instable). Réfutation : AUROC variance ≤ 0,5 + marge. MVP : sauvegarder moyenne et
  variance des per_position_logits sur les vues (ré-extraction PS/TM). Coût extraction ≈ cache E03.
- **E14 — gradients contrastés.** H : grad(accord vrai) − moyenne grad(accords faux) localise mieux que |grad|.
  Réfutation : AUROC ≤ 0,55. Coût : 1 backward × (1 + k contrôles) par image, PS/TM.
- **E15 — ré-embedding.** H : corrélation locale entre le résidu de ré-embedding (message connu) et le résidu de
  l'image (image − flou) est plus forte dans les zones marquées. Réfutation : AUROC ≤ 0,55 ou identique avec faux
  message. Coût : 1 embed/image ; risque : l'embedder dépend du contenu.
- **E16 — remover TrustMark.** H : |image − remover(image)| et la chute d'accord après retrait sont localement plus
  forts sur les zones marquées. Réfutation : AUROC ≤ 0,55 ou égal sur images non-TrustMark. Coût : 1 passe remover.
- **E17 — TruFor.** Bloqué (poids externes, licence, internet). Aucune action.
- **E18 — fusion.** Moyenne fixe des probabilités OOF de candidats confirmés dont la corrélation des erreurs par
  image est faible ; méta-modèle seulement avec cross-fitting imbriqué. Démarre après ≥ 2 confirmations.
