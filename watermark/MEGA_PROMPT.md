# Méga prompt : améliorer la localisation de watermark (team51, branche `autoresearch/nov21-wm`)

Tu es ingénieur de recherche en vision par ordinateur et PyTorch, dans une compétition qui se termine le
**22/11/2026 à 12:00 (heure locale)**. Ta mission : faire progresser la mIoU officielle de la tâche
« Watermark Localisation » au-delà du plateau actuel (~0,50), avec du code exécutable et des mesures honnêtes.
Lis tout ce prompt avant d'agir. Ne fabrique aucun résultat : chaque chiffre que tu annonces doit venir d'un log.

---

## 0. Règles non négociables

1. **CLAUDE.md du dépôt (cluster)** : la réservation `cispahack` est pleine, les nœuds sont exclusifs (4 GPU).
   **Toute l'équipe (`/p/scratch/training2666/delorme2/team51/`) ne peut avoir qu'UN job Slurm à la fois**
   (en cours + en attente). Vérifie avant chaque `sbatch` :
   ```bash
   T=/p/scratch/training2666/delorme2/team51
   squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "$T"
   ```
   Si quelque chose apparaît, l'équipe a déjà son nœud : ne soumets rien. Pour plus de travail, enchaîne avec
   `--dependency=afterany:<jobid>`. Jamais deux jobs. Soumets toujours depuis le dossier d'équipe.
2. Le propriétaire veut des jobs **courts et légers** (1 GPU, ≤ 1 h). Ne demande pas 4 GPU × 1 h 30.
3. Aucun calcul sur le nœud de login (téléchargements et lectures OK). Pas d'internet sur les nœuds GPU.
4. Ne modifie pas l'environnement partagé (`.venv`, `models/`, `gpu_job.sh`). Tes fichiers vont sous
   `work/jacquet2/` et `output/jacquet2/`.
5. Ne modifie jamais un fichier utilisé par un job en attente ou en cours sans le vouloir explicitement.
6. Clé API : uniquement dans `~/.watermark_api_key` (laptop et JURECA, chmod 600). Jamais dans le code, Git,
   les logs ou le chat.
7. Leaderboard : **délai d'environ 6 à 10 min entre deux soumissions** (HTTP 429 sinon). Seul le meilleur score
   public est conservé ; le classement final utilise 70 % cachés du test.
8. Pas de push GitHub ni d'envoi au leaderboard sans accord explicite (déjà accordé : « submit au fur et à mesure
   les meilleurs exp »).

---

## 1. Sources à auditer (dans cet ordre)

| Source | Où | À en retenir |
| --- | --- | --- |
| Énoncé officiel | `C:\Users\mathi\Downloads\Watermark Localization.pdf` (4 p.) | métrique, format, types d'éditions |
| README du kit | `$R/README.txt` | soumission ≤ 25 Mo selon le README, **20 Mo selon le PDF : garde 20** |
| Wrappers | `$R/wmloc/watermarks.py` | `features()`, `logits()`, résolutions internes |
| Données | `$R/wmloc/data.py`, `release/{train,test}/metadata.json`, `train/masks_128.npz` | champs disponibles |
| Métrique | `$R/wmloc/utils.py` (`downsample_mask`, `miou_image`, `score`) | **référence absolue** |
| Template | `$R/submission_template.py` | `BASE_URL=http://34.147.229.55`, `TASK_ID=34-watermark-localisation-image` |
| Code équipe | `hackaton/watermark/{baseline.py, run_all.py, README_baseline.md}` | baseline vérifiée |
| Autoresearch | `hackaton/watermark/autoresearch/{prepare.py, train.py, worker.py, multi_worker.py, program.md, results.tsv}` | boucle, historique |
| Plan d'équipe | `hackaton/watermark/TASK_PLAN.md`, `hackaton/TEAM_GUIDE.md`, `hackaton/CLAUDE.md` | contraintes |
| Papiers | PixelSeal arXiv 2512.16874 ; TrustMark ICCV 2025 ; StegaStamp CVPR 2020 ; Watermark Anything ICLR 2025 | mécanismes d'extraction |

`$R = /p/scratch/training2666/delorme2/team51/Hackathon_Watermark_Localisation_Task`

---

## 2. Faits vérifiés aujourd'hui (ne les re-découvre pas, mais re-vérifie si tu doutes)

### Tâche et données
- 3 méthodes : PixelSeal (256 bits), TrustMark Q (100 bits), StegaStamp (100 bits). **La méthode et le message
  sont fournis aussi au test.**
- Train 510 (170/méthode), test 1020 (340/méthode). Images 512×512, masques évalués en 128×128 (cellule 4×4, ≥ 50 %).
- Métrique : par image, moyenne des IoU des classes 0 et 1 (union vide → 1), puis moyenne sur les images.
- `full_watermark` (train seulement), 85/85 par méthode : **True** = image entièrement marquée puis une partie
  restaurée (masque ≈ 1 avec des trous) ; **False** = watermark seulement sur une partie (outpainting ou
  transplantation, masque ≈ 0 avec une zone à 1).
- Couverture **bimodale** : environ la moitié des images < 5 %, l'autre moitié > 90 %. Aucun masque vide ou plein.
- Aucun groupement par image source possible (noms = hashs, 510 messages tous distincts) → fuite possible
  entre variantes, à mentionner comme limite.
- **Ordres de grandeur des constantes** : tout-0 ≈ 0,49 sur les images à faible couverture et ≈ 0,03 sur celles
  à forte couverture ; tout-1 donne l'inverse. Un prédicteur « mode parfait + constante » vaut donc déjà
  **≈ 0,48**. Notre 0,50 n'est qu'à peine au-dessus : **les gains doivent venir de la localisation fine
  (petites zones, trous)**, pas du choix du mode.

### Extracteurs (wrappers officiels)
- PixelSeal : redimensionne l'image en 256 ; `convnext`, `embedding` et `per_position_logits` sont en **8×8**
  (pas de 32 px). Le logit global est la moyenne spatiale des logits par position.
- TrustMark : redimensionne en 245 ; ResNet-50 avec `layer1` 62×62, `layer2` 31×31, `layer3` 16×16, `layer4` et
  `per_position_logits` 8×8.
- StegaStamp : ONNX ; le wrapper n'expose que les logits globaux. Le graphe contient
  `gen_decoder/dec_conv_{1..7}/Relu:0` (dec_conv_3 100×100, dec_conv_5 50×50 à 400 px) : on les expose dans une
  **copie** du graphe (`work/jacquet2/models/stega_stamp_feats.onnx`). Pas de logits par position (couches
  denses après le flatten).
- **TrustMark n'a pas de poids dans le `.venv` partagé** et les nœuds GPU n'ont pas internet. Une copie privée
  du paquet avec les poids Q est dans `work/jacquet2/pylib/trustmark/` (`--pylib`).
- Signal observé : la **précision globale des bits** suit très bien la couverture (≈ 1,0 pour les images très
  marquées, ≈ 0,5 pour les peu marquées). La **carte d'accord par position** (Σ_k (2m_k−1)·l_k(h,w)/nbits) est
  faiblement positive à l'intérieur du masque, avec quelques images nettes (+1,6 / +2,1), et souvent ≈ 0.
- Shift-and-stitch ×4 (décalages circulaires du 512 d'entrée) densifie les cartes 8×8 en 32×32 pour PixelSeal
  et TrustMark (fait dans `baseline.py`).

### Infrastructure en place
- `prepare.py` : cache des features gelées (train et test, ~3,1 Go) dans `work/jacquet2/ar_cache/` ; 3 plis
  répartis par méthode × full_watermark (seed 0) ; `evaluate_oof` ; `write_submission` (validation complète).
- `train.py` : **le seul fichier modifié** par autoresearch ; tête par méthode ; `val_miou` = mIoU officielle
  sur les prédictions hors-pli (OOF) des 510 images ; seuil choisi sur l'OOF (un par méthode, léger optimisme
  documenté) ; `--final <path>` entraîne sur tout le train et écrit une soumission validée.
- `worker.py` : serveur d'expériences dans un job Slurm, file `work/jacquet2/ar_queue/<id>/{train.py,ARGS,READY}`,
  réservation atomique (`CLAIM`), `DONE` = code de sortie. `multi_worker.py` lance N workers dans UN job.
  Plusieurs workers peuvent partager un GPU (une expérience ≈ 2,7 Go, 40 à 160 s).
- Un `srun --overlap` dans un job à `gres/gpu:1` ne voit qu'**un** GPU utilisable, même si le nœud en a 4.
- `lb_submit.py` (`work/jacquet2/ar/`) envoie au leaderboard depuis le login et journalise dans
  `~/watermark_submissions.jsonl`.
- Incident : un « Disk quota exceeded » passager sur scratch a tué les workers (20h49). Rends les écritures du
  worker robustes (try/except sur le heartbeat, reprise).

### État de la recherche (OOF 3 plis, mIoU officielle)
| Exp | Changement | val_miou | Décision |
| --- | --- | --- | --- |
| ca313e5 | baseline (60 ep, lr 2e-3, BCE + Dice 2 classes, proj 32 / largeur 96, dilatations 1-2-4-8) | 0,4998 | référence |
| e01 | 120 epochs | 0,4977 | ❌ |
| e02 | Dice ×3 | 0,4973 | ❌ |
| e03 | tête plus petite (proj 16 / largeur 64) | 0,4998 | ✅ plus simple |
| e04 / e05 | lr 5e-3 / 1e-3 | 0,4968 / 0,4928 | ❌ |
| e06 | sans cartes de features (accord + RGB + scalaires) | 0,4717 | ❌ → les features comptent |
| **e07** | **moyenne de 3 seeds** | **0,5011** | ✅ **meilleur** |
| e08 / e09 | batch 8 / 30 epochs | 0,4990 / 0,4919 | ❌ |
| e10 | Dice seul | 0,4948 | ❌ |
| e12 | lissage 5×5 des probabilités | 0,4987 | ❌ |
| e13 / e14 / e15 | wd 1e-2 / dropout / 90 epochs | 0,4966 / 0,4982 / 0,4954 | ❌ |
| e16 | vecteur de contexte global | 0,4979 | ❌ |
| e17 | Lovász + BCE | 0,4968 | ❌ |
| e18 | grille 32 | crash (bug de taille dans la branche `up`) | à corriger |
| e21, e22, e25, e26 | 5 seeds, e07 + dropout, log1p, accord à 128 | **en cours** | — |

Par méthode (e07) : PixelSeal 0,478, TrustMark 0,536, StegaStamp 0,490.
**Leaderboard public : baseline 0,4978 (id 1501), e07 0,5064 (id 1532, meilleur).** L'OOF prédit bien le score
public (écart < 0,01).

**Conclusion : les réglages de la tête plafonnent. Le goulot, c'est l'information spatiale en entrée.**

---

## 3. Ce que je te demande

### Étape A : audit (30 min max, lecture seule)
1. Relis le PDF, `wmloc/*.py`, `baseline.py`, `prepare.py`, `train.py`, `results.tsv`. Signale toute
   incohérence avec la section 2 (par exemple une résolution, un champ de métadonnées, une règle de soumission).
2. Résume en 10 lignes, dans les papiers, **comment chaque décodeur agrège l'information spatiale** (pooling
   global, couches denses, STN de StegaStamp) et ce que cela implique pour la localisation.
3. Analyse les erreurs de e07 sur l'OOF : mIoU par méthode × full_watermark × tranche de couverture (< 5 %,
   5–50 %, 50–95 %, > 95 %). Où perd-on le plus ? (images à petite zone ? trous ? mauvais mode ?) Calcule aussi
   l'**oracle « mode parfait + constante »** et l'oracle « seuil par image ».

### Étape B : nouvelles sources de signal (priorité, classées)
Chaque idée = une expérience mesurée en OOF, une seule variable à la fois, avec son coût GPU mesuré.

1. **Décodage par occlusion / fenêtres** (le plus prometteur) : pour chaque image, on connaît le message.
   Masquer des régions (remplacer par du flou ou du gris, ou ne garder qu'une fenêtre) et mesurer la variation
   de l'accord soft avec le message au niveau global → carte d'importance par région. PixelSeal et TrustMark
   font une moyenne globale, donc masquer les zones propres doit **augmenter** l'accord. Grilles 8×8 ou 16×16,
   avec des fenêtres glissantes. Mesure d'abord le temps sur 16 images ; StegaStamp (ONNX, 400 px) est le plus
   lent : batch large, CUDA EP.
2. **Gradient de l'accord par rapport aux pixels** (PixelSeal et TrustMark sont différentiables en torch) :
   ∂(Σ(2m−1)·logit)/∂x, en valeur absolue, lissé → carte de saillance (grad×input, SmoothGrad léger).
3. **Cartes plus fines** : `layer1` de TrustMark (62×62), `dec_conv_1/2` de StegaStamp, shift-and-stitch ×8 ;
   calcul des features sur des tuiles à l'échelle native (attention à l'échelle du motif : PixelSeal embarque à
   256 puis agrandit le résidu, d'où la réduction 512→256 dans le wrapper).
4. **Résidu haute fréquence de l'image** (image − flou) en entrée de la tête : les watermarks post-hoc vivent
   dans les hautes fréquences ; les régions éditées ou restaurées ont des statistiques différentes.
5. **Ré-embedding** (les embedders sont fournis : `wm.embed(img, bits)`) : ré-embarquer le message connu et
   comparer le résidu au résidu estimé de l'image. À tester seulement si 1 et 2 échouent.

Ces nouvelles features modifient le cache : ajoute-les dans `prepare.py` (version v2, nouveau dossier de cache,
**sans écraser v1**), et documente le changement de harness (c'est une décision humaine, pas une tricherie de
la boucle).

### Étape C : tête et post-traitement (secondaire)
- Corrige e18 (grille 32 : la branche `up` attend la taille 64).
- Combine les quasi-gains : e07 (seeds) + ce qui ressort de e21 à e26.
- Seuil par mode : deux seuils par méthode selon la précision globale des bits (élevée ou faible), choisis
  sur l'OOF.
- Post-traitement par mode : remplissage des petits trous si l'image est prédite majoritairement marquée,
  suppression des petites composantes sinon (taille minimale choisie sur l'OOF). **Pas de forme rectangulaire
  imposée** (trous et transplantations).
- Garde la règle de simplicité d'autoresearch : un gain < 0,002 avec beaucoup de code = refus.

### Étape D : fiabilité
- Rends `worker.py` robuste aux erreurs d'écriture passagères (quota) : réessaie, ne meurs pas.
- Évite le surapprentissage de l'OOF : pour toute « victoire » > 0,003, relance avec une autre seed de plis
  (`SEED` de `prepare.py` dans une variante) avant de soumettre.
- Mesure le temps et la mémoire GPU de chaque expérience ; budget ≤ 5 min (kill à 10 min).

---

## 4. Boucle d'exécution (autoresearch adaptée)

```text
pour chaque idée :
  1. variante de train.py (ou prepare v2) depuis le meilleur commit → git commit sur autoresearch/nov21-wm
  2. upload dans work/jacquet2/ar_queue/<id>/train.py (+ ARGS), puis touch READY
  3. lire run.log : grep "^val_miou:"   (crash → tail -n 40 run.log, correction triviale ou "crash")
  4. ligne dans results.tsv (commit, val_miou, mémoire Go, keep/discard/crash, description)
  5. si val_miou augmente strictement → keep ; sinon git reset --hard HEAD~1
  6. si nouveau meilleur ET gain ≥ 0,002 : ARGS "--final output/jacquet2/ar/sub_<commit>.pt" → vérifier →
     python work/jacquet2/ar/lb_submit.py <fichier> <label>   (respecter le délai, ~6-10 min)
```
- Remplis toujours la file pour que les workers ne soient jamais inactifs (au moins 2 à 4 expériences en avance).
- Un seul job Slurm pour l'équipe ; s'il se termine, enchaîne le suivant avec `afterany`
  (1 GPU, ≤ 1 h, `multi_worker.py --per-gpu 2..4`).
- Garde une marge : **dernière soumission avant 11:30 le 22/11**, avec la meilleure version re-générée et
  re-validée (clés, 1020 noms uniques, forme [1020,128,128], bool, ≤ 20 Mo).

---

## 5. Livrables attendus à chaque point d'étape
1. Fichiers créés ou modifiés (chemins).
2. Expériences lancées : id, changement, val_miou, temps, mémoire, décision.
3. Soumissions : id, label, score public, delta.
4. Ce qui a réellement été exécuté et ce qui reste à faire.
5. Limites (optimisme du seuil sur l'OOF, absence de groupement par source, variance entre seeds) et prochain
   test le plus utile.

Ne dis jamais qu'un entraînement a tourné s'il n'a pas tourné. Si un fichier indispensable manque, nomme-le
précisément et avance sur les parties indépendantes.
