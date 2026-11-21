# New localisation ideas (scout pass, 21 Nov 2026)

Scope: ideas that are not in SOLUTION_LANDSCAPE.md E01–E18, or that make one of them a specific, cheap mechanism (the overlap is marked "sharpens Exx").
Labels: **FACT** = read in the cited source. **HYP** = my hypothesis, untested. Nothing was run.
Costs assume about 1 A100, 1-hour jobs and about 13 h left. "AUROC probe" means the tie-aware per-image AUROC against the 128 mask that is already in use, on the train split.

## Top 8 (ranked by expected value / cost)

| ID | Idea | Mechanism (1 line) | Cost | Risk | EV | Smallest falsifying test |
|---|---|---|---|---|---|---|
| N01 | Expected-mIoU decoding (Nowozin) | For each image, pick the top-k cells by calibrated p that maximise the *expected* 2-class mIoU, instead of a fixed threshold | CPU, <1 h, no retraining | Low | High: targets the 0.508→0.535 per-image-threshold gap and tiny regions | Run on the existing OOF probs (temperature fitted in nested CV). Kill if < +0.003 |
| N02 | Local residual regression vs predicted residual | r̂ = embed(x,m)−x. Map α(p) = ⟨HP x, HP r̂⟩_win / ⟨HP r̂,HP r̂⟩_win, minus the same with a wrong message m' | 2 embed calls per image, about 20 min for all | Med | High: pixel-resolution evidence; the only cue likely to work for StegaStamp | AUROC probe on 30 imgs/method of α(m)−α(m'). Kill if < 0.62 |
| N03 | Exact StegaStamp attribution at the flatten layer | logit_k = W2 diag(g) (W1 f + b1) + b2 is exactly linear in f once the ReLU gates g are fixed → per-position contribution map 13×13, signed by the message, then unwarped by the STN | ONNX weights + the existing dec_conv_7, about 30 min | Low-Med | Med-High: StegaStamp gives up 1/3 of the score at AUROC 0.52 | AUROC probe of the signed 13×13 contribution map (after STN inverse) |
| N04 | Rectangle / row-column structure prior | Low-coverage regime: target is often an axis-aligned rectangle (outpainting, original placement). Fit the best rectangle on p with integral images. Detect rectangle edges with 1-D changepoints on row/column agreement profiles | CPU, 1–2 h | Med | Med-High on the full_wm=False half | First the oracle: fraction of False masks with area/bbox > 0.9, and mIoU of GT-bbox. Then the fitted rectangle vs free mask on OOF |
| N05 | Random-mask linear regression (Kernel-SHAP style) | 256 random 50% masks with blur infill. Regress raw per-bit signed logits on the masks (ridge) → 16×16 contribution map. Beats single-cell occlusion, which redundancy hides (sharpens E06/S6) | about 400k extractor passes ≈ 10–20 min (ONNX for StegaStamp: about 45 min) | Med | Med | AUROC probe vs the del8 occlusion (0.58 PixelSeal) on 60 imgs/method |
| N06 | Message-swap self-supervision from train images | Re-embed a *new* message m' on random train-shaped regions of train images. Target = region of m'. Only message-conditioned channels are computed w.r.t. m'. Unlimited labelled pairs, no clean source needed (sharpens E08) | Embedding + caching about 1 h, pretrain + finetune head | High | Med | Pretrain on synthetic, finetune on real folds. Paired OOF delta with 1 seed |
| N07 | Resampling / periodicity forensic maps | All three embedders add a residual made at low resolution (TrustMark/PixelSeal 256→512 ×2, StegaStamp 400→512 ×1.28). Local spectral peaks of the 2nd-difference p-map (Popescu–Farid) show the upsampled residual. VAE period-8 peaks show inpainted areas | CPU/GPU FFT, <1 h | Med | Med (StegaStamp, holes) | AUROC probe of local peak energy at the method-specific frequency, 30 imgs/method |
| N08 | Decoder-verified per-image threshold choice | For K candidate thresholds, decode x_M = x⊙M+blur⊙(1−M) and x_¬M. Choose the M on the "A(x_M) high, A(x_¬M) low" plateau that has the smallest area | 2K passes per image, minutes | Med-High | Med | OOF: mIoU of the selected threshold vs the global pair of thresholds |

Suggested order: N01 (now, CPU) ‖ N03 + N02 probes (GPU, short) → N04 oracle → N07 probe → N05 → N06/N08 only if time remains.

---

## N01 — Expected-mIoU (Bayes-optimal) decoding  [new; complements the threshold work]
- **Mechanism (FACT, Nowozin, "Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case", CVPR 2014, §3–4, https://openaccess.thecvf.com/content_cvpr_2014/papers/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.pdf):** with factorised marginals p_i, the IoU-optimal labelling is a top-k set by p. The expected IoU is approximated by E[I]/E[U], and k is found by a linear scan over the sorted p.
- **Adaptation (HYP):** our score is the mean of the class-1 and class-0 IoUs, with empty union = 1. For each image, sort p in descending order. For k = 0..16384: E[I1]=Σ_{top k} p, E[U1]=k+Σ_{rest} p, E[I0]=Σ_{rest}(1−p), E[U0]=(N−k)+Σ_{top}(1−p). Score(k)=½(E[I1]/E[U1]+E[I0]/E[U0]), with exact handling at k=0 and k=N. O(N log N). The decoder predicts all-zero when evidence is diffuse, which is the tiny-region failure mode.
- **Needs calibration:** fit a temperature, or isotonic regression per method × bit-acc regime, on inner folds only, then apply to the outer fold.
- **Why localisation:** it turns the per-image-threshold oracle (0.535) into a label-free rule. It uses each image's own probability mass to set its coverage.
- **Falsify:** on the existing OOF probs, compare against the two-threshold baseline (paired per-image delta, bootstrap). Kill if < +0.003.
- **Variants:** (a) blend Score(k) with a prior on k from the regime; (b) Monte-Carlo version that samples masks from p with spatial correlation (sample a low-res Gaussian field). Spatially correlated errors make the factorised estimate over-confident.
- **Risk:** low. Miscalibration within an image (spatially correlated errors) can bias k. Mitigation: calibrate on 32×32 pooled cells.

## N02 — Local regression on the embedder's predicted residual  [sharpens E15 into a pixel-level test]
- **FACT (TrustMark code, `encode()`, https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py):** the image is resized to 256 (bilinear) and encoded. residual = clamp(stego)−cover, minus its per-channel mean, bilinearly upsampled to the crop size, scaled by WM_STRENGTH, added and clipped. The embedder is deterministic given (image, message).
- **HYP:** embedding is near-idempotent in the residual. For x = x0 + r_true(x0,m), r̂ = embed(x,m)−x ≈ c·r_true locally (c may be <1 if the network adds less on an already-marked image; even c≠1 keeps the correlation). In watermarked areas x contains r_true, so the local least-squares coefficient α(p)=⟨HP(x),HP(r̂)⟩/⟨HP(r̂),HP(r̂)⟩ over a 16–32 px window is >0. In clean or inpainted areas it is ≈0. Content-adaptive (JND-type) residual components correlate with texture whatever the message. Remove them with the differential r̂(m)−r̂(m'), or by subtracting α computed with 2–4 random messages m'.
- **Variants:** (i) for StegaStamp, use a band-pass matched to the 400-px residual band; (ii) also store the energy-reduction ratio ‖HP(x−r̂)‖²_win/‖HP(x)‖²_win; (iii) pooled to 128×128, this is a new head input channel at native resolution, about 8× finer than the per-position logits.
- **Why localisation:** this is evidence at the pixel level and does not depend on the extractor's receptive field. Edges of holes and transplanted objects become sharp.
- **Falsify (≈15 min):** 30 images per method. AUROC probe of α(m)−mean α(m'). Kill per method if < 0.62. Positive control: an image freshly embedded on a known rectangle.
- **Risk:** the residual may not be reproducible on already-marked or edited content (PixelSeal UNet global context, StegaStamp at 400 px). Image saving or quantisation may wash out a weak residual (PixelSeal is very imperceptible). PNG vs JPEG storage is unknown: check.

## N03 — Exact flatten-layer attribution for StegaStamp  [new; complements E04]
- **FACT (official decoder, https://github.com/tancik/StegaStamp/blob/master/models.py, `StegaStampDecoder`):** STN (3 strided convs → Dense128 → affine 6, initialised to identity) → warp. Then 7 convs (32/s2, 32, 64/s2, 64, 64/s2, 128/s2, 128/s2) → Flatten → Dense(512, ReLU) → Dense(100), raw logits. A 400 input gives 13×13×128 (12.5 rounded, `same` padding: verify in the ONNX).
- **Mechanism (exact, not a gradient on pixels):** let f = flatten(dec_conv_7), h = W1 f + b1, gates g = 1[h>0]. Then logit = W2 diag(g) W1 f + const. The contribution of position (i,j) to bit k is c_k(i,j) = Σ_c [W2 diag(g) W1]_{k,(i,j,c)} f_{ijc}. The agreement map is A(i,j)=Σ_k s_k c_k(i,j). This is the StegaStamp analogue of the per_position_logits of the two other methods, which we currently lack. The dense layer gives position-specific weights, so StegaStamp does *not* pool spatially. That plausibly explains why occlusion there is ≈ chance.
- **Then:** map (i,j) back through the inverse STN affine (already being worked on) into a 32×32 grid, with a validity mask for cells outside the warped field.
- **Assets:** read W1/W2/b with `onnx.numpy_helper` over the graph initialisers. Get the per-image gates from the ONNX Dense output, or recompute in numpy. Check the NHWC vs NCHW flatten order: TF exports flatten in HWC order.
- **Falsify:** AUROC probe of A (unwarped) vs mask. Sanity check: Σ_ij c_k(i,j) + const = logit_k exactly.
- **Risk:** the dense weights may encode bits spread over the whole image (each bit read everywhere). Then A is informative. If instead bits are read at fixed positions, A is noisy per position but still reflects local presence. Low cost, so test it.

## N04 — Rectangle / outpainting structure priors and 1-D profiles  [new]
- **FACT (task contract):** the original watermark sits on a rectangle (aspect ≤16:1, or ≤2:1 for TrustMark). Outpainting keeps the watermark only on the original part, so the target is axis-aligned. Removal/addition cut holes; transplantation gives an irregular object.
- **HYP:** in the full_wm=False regime, a large share of masks are (rectangle − holes). An axis-aligned rectangle fit is far more robust than per-cell thresholding: summing evidence along 128-cell lines boosts SNR by about √128.
- **Mechanism:** (a) row/column profiles R(y)=mean_x p(x,y), C(x). Find 2 changepoints each (exhaustive search over pairs on a 128 grid, or 1-D two-step fit). (b) Exact search of the best rectangle on a 32×32 grid with integral images (32^4/4 ≈ 260k candidates; vectorisable) under the aspect limit for the method. Expected mIoU is computed with N01's formula. (c) Final mask = rectangle ∧ (p > low threshold), to keep holes. (d) Selector (rectangle vs free mask) from fit quality: fraction of p-mass inside the rectangle, plus the regime scalar.
- **Falsify:** step 0 (5 min, CPU): rectangularity of the train masks per regime and the mIoU of the GT-bbox oracle. If the bbox oracle is not well above 0.6 on False images, drop it. Step 1: fitted rectangle on OOF probs vs current output, False regime only.
- **Also:** in the True regime, outpainting holes form a *frame* (border strips). Run the same 1-D profile test on the complement.
- **Risk:** transplantation and irregular shapes. Hence the selector, and never forcing the fit.

## N05 — Random masks + linear regression  [sharpens E06 / RISE S6]
- **FACT (RISE, Petsiuk et al., BMVC 2018, §3, https://arxiv.org/abs/1806.07421):** saliency = expectation of the score over random masks. **FACT (KernelSHAP, Lundberg & Lee, NeurIPS 2017, §4, https://arxiv.org/abs/1705.07874):** a weighted linear regression of model outputs on binary masks gives additive attributions.
- **HYP / why it beats our occlusion:** watermarks are redundant. Removing one 64-px cell barely moves decoding (AUROC 0.58), but removing about 50% changes it strongly. Regression over M masks recovers each cell's marginal contribution, with interactions averaged out. Regress *raw per-bit signed logits* s_k·l_k (not the saturated soft agreement), summed over k, or ridge per bit and then summed.
- **Setup:** grid 16×16 (32 px cells), M=256 masks with p=0.5 smoothed by bilinear upsampling (RISE style). Infill = Gaussian blur σ≈4, which already removes high frequencies; option: TrustMark remover output as infill (E16 assets). Ridge λ chosen once on a few train images.
- **Cost:** 1530 imgs × 256 × 3 methods. PixelSeal/TrustMark at 256/245 px, batched: minutes. StegaStamp ONNX at 400 px is the bottleneck: estimate the throughput first; use M=128 if needed.
- **Falsify:** AUROC probe on 60 imgs/method vs del8. Keep only if it beats del8 by ≥0.03 for a method.

## N06 — Message-swap self-supervised pairs from the train images  [sharpens E08]
- **Mechanism (HYP):** take a train image x (any regime) and a fresh random message m'. Embed x' = embed(x, m'), paste x' into x through a synthetic mask S (sampled from *other* train masks, rectangles, objects; feathered or not). Compute the message-conditioned channels (agreement with m', N02/N03 maps with m') and set target = S. The pre-existing watermark with m acts as realistic interference with another message. No clean source is needed, and the targets are exact.
- **Head:** pretrain only on message-conditioned and RGB channels. Message-agnostic features (raw convnext/layerN) would also detect the old watermark, so exclude them from pretraining or zero them, then finetune on the real folds.
- **Leakage:** build synthetics only from that fold's training images. Never from the outer validation fold, never from test (transductive use of test pixels is a rules question: avoid).
- **Falsify:** 1 seed, paired OOF delta. Kill if ≤ 0.
- **Risk:** the domain gap is large. Real holes come from generative inpainting, synthetic ones are pure pastes. Double-watermark interaction. Highest cost of the top 8.

## N07 — Resampling-trace and periodicity forensics  [new; distinct from E05 generic filters]
- **FACT:** TrustMark residual is made at 256 and bilinearly upsampled (code above). StegaStamp works at 400 px (contract), so the residual or the image is resampled to 512 (whether by the WAVES pipeline or by the challenge is unverified). **FACT (Popescu & Farid, "Exposing digital forgeries by detecting traces of re-sampling", IEEE TSP 2005, §2–3, https://farid.berkeley.edu/downloads/publications/sp05.pdf):** interpolation creates periodic correlations between neighbouring pixels. These show up as peaks in the spectrum of the local linear-predictor residual (p-map). Peak position depends on the resampling factor (×2 vs ×1.28 → different frequencies).
- **HYP:** a ×2 or ×1.28 interpolated residual of meaningful amplitude leaves a localisable peak in watermarked areas only. StegaStamp's residual is visible to the eye, so it is the best candidate. Generative inpainting (latent-diffusion VAE, factor 8) leaves period-8 peaks in *edited* areas. That is a hole detector for the True regime and for outpainting frames.
- **Cheap version:** take a fixed 2nd-difference / Laplacian filter, square it, then compute a 64×64 sliding FFT with a 16-px stride. Features = energy at the expected peaks (1/1.28 aliasing frequencies for StegaStamp, 0.5 cycles/px for ×2, 1/8 for VAE). Pool to 32×32 and probe.
- **Falsify:** AUROC probe per method and per frequency, 30 imgs each. Kill any frequency below 0.6.
- **Risk:** the peak is masked if the whole image (including the non-watermarked parts) was resampled. Then the cue is uninformative. The probe settles this in 20 min.

## N08 — Decoder-verified threshold choice at test time  [new]
- **Mechanism (HYP):** for the head probability map, take K≈7 candidate masks M_t (thresholds or N01 top-k levels). Compute A_in(t) = agreement(x⊙M_t + blur(x)⊙(1−M_t)) and A_out(t) = agreement(x⊙(1−M_t) + blur⊙M_t). As the mask grows past the truth, A_out drops to chance and A_in stays high. Choose the smallest M_t with A_out(t) ≤ chance+δ and A_in(t) ≥ A_in(max)−ε. δ and ε are tuned on the train OOF, 2 parameters per method.
- **Why localisation:** this uses the decoder as a verifier of the *extent*, the dimension where we lose the most (per-image-threshold oracle). It also fixes regime errors cheaply (7/510).
- **Falsify:** on OOF, oracle-free selection vs the current two thresholds. Kill if < +0.003.
- **Risk:** redundancy. Small residual watermark islands outside M keep A_out high. Blur infill may not fully erase the watermark (check A on a fully blurred image = chance). StegaStamp decoding with the STN is less reliable on composites.

---

## Further ideas (lower EV or cost too uncertain)

- **N09 JND/texture "expected-evidence" channel.** FACT: PixelSeal attenuates the residual with a JND map (Pixel Seal arXiv 2512.16874, §3; videoseal repo `videoseal/modules/jnd.py`). HYP: low agreement in flat areas is weak evidence of absence. Feed the JND map (or local variance at 256 px) as an input so the head learns a confidence-weighted reading. Cost: tiny. Test: add a channel, 1 seed, paired delta. Targets false negatives in flat regions.
- **N10 Per-position logit *variance across shifts* as a phase-consistency channel.** You already compute 16 circular shifts for shift-and-stitch. Store the per-cell std across the 16 shifts and their sign agreement, not just the interleaved values. Watermarked cells should give stable signed evidence; clean cells noise. Cost: zero extra passes. Overlaps E13, but this is the minimal form.
- **N11 Bit-split agreement maps.** Split the 256/100 bits into 4 random disjoint groups. Compute 4 agreement maps and give their min/median to the head. A clean cell rarely agrees on all groups by chance; texture-driven bias tends to hit groups coherently in a message-independent way. Overlaps E03; cheap form with no extra passes.
- **N12 Coverage regression → k prior.** Predict per-image coverage directly (small regressor on scalars + pooled maps; train has the truth). Use it as the k prior for N01/N04. The regime is already right 503/510, so the remaining error is coverage *within* a regime. Test: R² of OOF coverage, then N01 with the prior.
- **N13 TrustMark remover as infill and as a differential.** FACT: `remove_watermark()` resizes to 256, predicts a residual, upsamples it, and pastes it back (TrustMark code). HYP: |x−remove(x)|, high-passed and locally pooled, is larger where the TrustMark residual sits. Use it as an N05 infill for TrustMark images. Overlaps E16; use it as an infill and as a channel, not a mask.
- **N14 Two-pass "decode the complement".** After the first prediction, blur the predicted watermarked region and re-run the extractors. Residual per-position agreement in the predicted-clean area reveals missed watermark (false negatives in large holes). A cheap second-stage input. Risk: blur edges leak.
- **N15 Model-free baselines worth a row in results.tsv:** (a) per-method thresholded agreement map upsampled (no head); (b) N04 rectangle on raw agreement; (c) all-ones/all-zeros chosen by bit acc (≈0.455 constant oracle). These set the floor that any head must beat. N04(b) may be surprisingly competitive in the False regime.

## Calibration / thresholds: rules that avoid overfitting
- Fit any temperature, isotonic map or (δ, ε) on **inner** folds (2/3 of each outer train fold) and apply to the outer fold. Report the outer mIoU only.
- Prefer rules with ≤2 parameters per method (N01 has 0 after calibration, N08 has 2). Count every rule tried in the run log.
- Report the paired per-image delta with a stratified bootstrap CI. The leaderboard (0.5097) has ~1020 images, about ±0.01 noise, so do not chase it.

## Facts vs hypotheses: summary
- Facts used: TrustMark residual pipeline and remover (adobe/trustmark code); StegaStamp decoder architecture (tancik/StegaStamp models.py); Nowozin's IoU decision theory; RISE; KernelSHAP; Popescu–Farid resampling detection; task contract.
- Unverified: whether the challenge images are PNG or lossy; whether StegaStamp images had their *whole* content resampled 512↔400; the NHWC flatten order in the ONNX; whether re-embedding is near-idempotent (N02); rectangularity of False masks (N04).
