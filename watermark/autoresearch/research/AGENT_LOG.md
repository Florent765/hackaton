# Agent log (A## experiments, 21-22 Nov 2026)

Baseline: N03_attr (confirm 0.5200, 3 seeds) and N03+N01 (0.5231). Protocol: 1-seed filter vs N03_attr seed 0
(paired, compare.py), promotion if confirm delta > +0.003 and CI low > -0.002, confirmation with seeds 0,1,2.
All numbers below were read from the run logs in /p/project1/training2666/jacquet2/ar_queue/<task>/run.log.

## A01 — finer exact StegaStamp attribution (dec_conv_6 / 5 / 4) and shift-TTA
- Hypothesis: N03's exact decomposition at dec_conv_7 (13x13) is resolution-limited. With the dense gates fixed the
  decoder tail is piecewise linear, so back-propagating the same objective <flatten(c7), u> through dec_conv_7/6/5
  gives exact grad x input decompositions at 25x25 / 50x50 / 100x100 (up to conv-bias terms).
- Code: A01/probe_graph.py (task A01_probe: conv attributes), A01/prepare_deep.py (A01_prep), A01/exp.py.
  Torch re-implementation of dec_conv_5..7 matches ONNX exactly (max abs diff 0.0); a7 reproduces N03 (corr 1.0000).
- Probe (train, per-image AUROC, true / random message): a7 0.695/0.502, a6 0.662/0.503, a5 0.594/0.503,
  a4 0.527/0.501, shift-TTA of a7 (4x4 x 8 px) 0.624/0.524. -> deeper = less AUROC, TTA worse; tested a6, a6+a5.
- A01_a6 (N03 + a6 channel), seed 0 (A01_a6_s0, CMP_A01_a6_s0): confirm 0.5195 vs 0.5162, delta +0.0032
  CI95 [+0.0008,+0.0057], W/L 188/147; explor +0.0040; fixed05 +0.0051; by method confirm ps +0.0001 ss +0.0085 tm +0.0011.
- A01_a6 seeds 0,1,2 (A01_a6_s1/_s2, CMP_A01_a6_s012): confirm 0.5200 -> 0.5242, delta +0.0042 CI95 [+0.0016,+0.0068],
  W/L 203/171; explor 0.5222 -> 0.5267 (+0.0045 [+0.0023,+0.0070]); fixed05 0.5174 -> 0.5227 (+0.0053);
  by method confirm ps -0.0025 ss +0.0150 tm +0.0002. **Confirmed** (vs N03_attr).
- Control A01_a6_ctrl (dec_conv_6 map with a random message), seed 0 (A01_a6_ctrl_s0): confirm 0.5156 (vs N03 0.5162,
  vs A01_a6 0.5195) -> the gain needs the true message.
- A01_a65 (N03 + a6 + a5): seeds 0/1/2 confirm 0.5213 / 0.5187 / 0.5168 (A01_a65_s0..s2). 3 seeds vs N03_attr
  (CMP_A01_a65_s012): confirm 0.5200 -> 0.5249, delta +0.0048 CI95 [+0.0020,+0.0076], W/L 213/176; explor 0.5222 -> 0.5272
  (+0.0050 [+0.0021,+0.0079]); by method confirm ps -0.0009 ss +0.0164 tm -0.0010 (PS/TM inputs unchanged: GPU noise).
  Same as A01_a6 within noise; a65 used as the StegaStamp part of the combined model.
- Decision: **keep** (StegaStamp +0.015/+0.016, message-specific).

## A02 — rectangle prior for the low-coverage regime (N04 x N01)
- Oracle (A02_oracle, labels used only for diagnosis): lo regime (bit acc <= 0.75) has ~0.53 of masks filling > 98% of
  their bbox, mean coverage 0.07-0.09; GT-bbox oracle mIoU 0.86-0.89 vs N03 confirm 0.49. Hi regime: complement is not
  rectangular (fill median 0.42-0.55).
- Method (A02/eval_rect.py, task A02_rect_N03 on N03_attr seeds 0,1,2): calibrated q (N01 grid, inner-fold
  calibration), exhaustive rectangle search on a 32x32 grid maximising expected mIoU; lo images -> rectangle,
  hi images -> N01 top-k.
- Result: rect confirm 0.5231 vs N01 0.5231 (delta -0.0000, CI [-0.0028,+0.0028]); lo-regime only +0.0025.
  The 'sel' policy (rectangle if its expected mIoU beats top-k) never fires: top-k is the unconstrained optimum of the
  same factorised objective (design error, no information). **Rejected**: the bottleneck is the probability map in the
  lo regime, not the shape decoding.

## A03 — differential re-embedding matched filter (N02, message-differential form)
- Map: per 16x16 block, normalised <HP(x), embed(x,m) - embed(x,m')>; control: two random messages. 48 train images/method
  (A03_probe). AUROC true / ctrl: PixelSeal 0.491/0.489, TrustMark 0.516/0.512, StegaStamp 0.542/0.503
  (message-independent raw residual for StegaStamp 0.630). **Rejected** (no message-specific signal; residuals are not
  reproducible at the whole-image scale).

## A04 — multi-scale crop decoding (the decoder as a sliding verifier)  [new idea]
- Hypothesis: in the lo regime the watermark sits on a rectangle that was embedded at its own scale (patch resized to
  the embedder resolution). A crop matching that rectangle, resized to the decoder resolution, decodes the known message
  far better than the whole image; scanning crops therefore localises the rectangle.
- Probe (A04/probe_crops.py, task A04_probe; 24 lo-flag train images/method, 645 square windows 96..352 px, stride s/4):
  TrustMark full-image BA 0.526, max crop BA 0.722 (ctrl 0.649), several images 0.95-0.99, best-crop IoU 0.27
  (0.6-0.76 on 10/24), AUROC of max-over-covering-crops map 0.671 (ctrl 0.534; existing agree map 0.586).
  PixelSeal: max BA 0.622 (ctrl 0.597), AUROC 0.578 (ctrl 0.505). StegaStamp: AUROC 0.561 vs ctrl 0.597 -> not used.
- Code: A04/prepare_crops.py (tasks A04_prep_tm_train 215 s, A04_prep_ps_train 256 s): 5 channels per image (max over
  covering crops of mean_k tanh(s_k l_k / 4), one per window size) at 32x32, true and random message. A04/exp.py adds
  them (x4, no standardisation) to the agreement stack of PixelSeal and TrustMark, on top of N03.
- A04_crops seed 0 (A04_crops_s0): confirm 0.5346 (explor 0.5350, fixed05 0.5318) vs N03_attr s0 0.5162.
  Control A04_crops_ctrl seed 0 (random message): confirm 0.5171 -> gain is message-specific.
- A04_crops seeds 0,1,2 (s1 0.5270, s2 0.5273; CMP_A04_crops_s012): confirm 0.5200 -> 0.5377, delta +0.0177
  CI95 [+0.0116,+0.0242], W/L 249/195; explor 0.5222 -> 0.5395 (+0.0173 [+0.0114,+0.0234]); fixed05 0.5174 -> 0.5326
  (+0.0152 [+0.0088,+0.0217]); by method confirm ps +0.0038 ss +0.0003 tm +0.0489; by coverage [0,.05) +0.023,
  [.05,.5) +0.042, [.5,.95) +0.005, [.95,1] +0.001. **Confirmed.**
- A04_crops_a65 (crops for PS/TM + A01_a65 maps for SS), seeds 0/1/2 confirm 0.5373 / 0.5318 / 0.5319.
