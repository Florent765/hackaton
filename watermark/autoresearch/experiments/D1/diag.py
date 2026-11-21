"""D1 diagnostics (preconditions of E01 / E03 / E04, coverage audit, tie-aware AUROC). Run on a GPU worker.
No training. Prints facts only."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from scipy.stats import rankdata  # noqa: E402


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)                                   # average ranks for ties
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def up128(m):
    return F.interpolate(m[None, None].float(), size=128, mode="bilinear", align_corners=False)[0, 0].numpy()


print("== sanity: tie-aware AUROC of a constant map =", auroc(np.zeros(100), np.r_[np.zeros(50), np.ones(50)]))

# ---------------------------------------------------------------- coverage audit
print("\n== coverage audit (train masks_128)")
for m in P.METHODS:
    c = P.load_cache(m); cov = c["masks"].float().flatten(1).mean(1).numpy(); fw = c["full_watermark"].numpy()
    h = np.histogram(cov, bins=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0001])[0]
    print(f"{m}: hist {h.tolist()} | fw=False mean {cov[~fw].mean():.3f} max {cov[~fw].max():.3f} | "
          f"fw=True mean {cov[fw].mean():.3f} min {cov[fw].min():.3f} | empty {(cov == 0).sum()} full {(cov == 1).sum()}")

# ---------------------------------------------------------------- E01: negative activations
print("\n== E01: sign of cached extractor maps (train)")
for m in P.METHODS:
    c = P.load_cache(m)
    for n in c["feature_names"]:
        x = c["feats"][n].float()
        q = torch.quantile(x.flatten()[torch.randperm(x.numel())[:2_000_000]], torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
        print(f"{m:10s} {n:34s} shape {tuple(x.shape[1:])} neg_frac {float((x < 0).float().mean()):.4f} "
              f"min {float(x.min()):.3f} q {np.round(q.numpy(), 3).tolist()} max {float(x.max()):.3f}")

# ---------------------------------------------------------------- tie-aware AUROC of existing maps, all train
print("\n== per-image AUROC (ties averaged), all train images with both classes")
for m in P.METHODS:
    c = P.load_cache(m); masks = c["masks"].numpy(); res = {}
    cands = {}
    if "agree" in c["feats"]:
        cands["agree_logit"] = c["feats"]["agree"][:, 0].float(); cands["agree_sign"] = c["feats"]["agree"][:, 1].float()
    for sc in ("occ", "occ16"):
        try:
            cands[sc] = R.load_sidecar(m, "train", sc, c["names"])[:, 0]
        except FileNotFoundError:
            pass
    for k, v in cands.items():
        a = [auroc(up128(v[i]), masks[i]) for i in range(len(masks))]
        res[k] = round(float(np.nanmean(a)), 4)
    print(m, res)

# ---------------------------------------------------------------- E03: local logits and messages
print("\n== E03: local logits / messages")
import json  # noqa: E402
meta = json.loads((P.TASK_ROOT / "release/train/metadata.json").read_text())
for m in P.METHODS:
    msgs = np.array([[int(b) for b in x["message"]] for x in meta if x["method"] == m])
    corr = np.corrcoef(msgs.T); off = corr[~np.eye(len(corr), dtype=bool)]
    print(f"{m}: K={msgs.shape[1]} mean bit {msgs.mean():.3f} | |inter-bit corr| mean {np.nanmean(np.abs(off)):.3f} "
          f"max {np.nanmax(np.abs(off)):.3f} (random ~{1 / np.sqrt(len(msgs)):.3f})")
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402
ds = LocalisationDataset(P.TASK_ROOT / "release", "train")
for m in ("pixelseal", "trustmark"):
    wm = load_watermark(m, P.TASK_ROOT / "models", "cuda")
    idx = [i for i, x in enumerate(ds.meta) if x["method"] == m][:4]
    x = torch.stack([torch.from_numpy(ds[i]["image"]).permute(2, 0, 1) for i in idx]).float().div(255).cuda()
    f = wm.features(x); pp = f["per_position_logits"]; g = wm.logits(x)
    print(f"{m}: per_position_logits {tuple(pp.shape)} {pp.dtype}; global logits {tuple(g.shape)}; "
          f"max|mean_hw(pp) - global| = {float((pp.mean((2, 3)) - g).abs().max()):.4g}")
    del wm; torch.cuda.empty_cache()

# ---------------------------------------------------------------- E04: StegaStamp graph geometry
print("\n== E04: StegaStamp ONNX geometry")
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
g = onnx.load(str(P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx")).graph
prod = {o: n for n in g.node for o in n.output}
first = next(n for n in g.node if n.output[0].startswith("gen_decoder/dec_conv_1/BiasAdd"))
print("dec_conv_1 first node:", first.op_type, first.name, "inputs", list(first.input))
# walk back from the decoder's first conv to the 'image' input, collecting op types
seen, stack, ops, path_nodes = set(), [first.input[0]], {}, []
while stack:
    t = stack.pop()
    if t in seen or t not in prod:
        continue
    seen.add(t); n = prod[t]; ops[n.op_type] = ops.get(n.op_type, 0) + 1; path_nodes.append(n)
    stack += [i for i in n.input if i]
print("ops upstream of dec_conv_1:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
names = [n.output[0] for n in path_nodes]
stn_like = [x for x in names if any(k in x.lower() for k in ("stn", "transform", "theta", "spatial", "loc"))]
print("STN-like tensor names upstream (first 25):", stn_like[:25])
mm = [n.output[0] for n in path_nodes if n.op_type in ("MatMul", "Gemm")]
print("MatMul/Gemm upstream:", mm)
# expose the decoder-conv input and the dense outputs, compare with the input image on real images
model = onnx.load(str(P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx"))
extra = [first.input[0]] + mm
for t in extra:
    vi = onnx.helper.make_empty_tensor_value_info(t); vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    model.graph.output.append(vi)
tmp = Path("/p/project1/training2666/jacquet2/research/D1"); tmp.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(tmp / "stega_geom.onnx"))
sess = ort.InferenceSession(str(tmp / "stega_geom.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ONNX providers:", sess.get_providers())
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cuda")
idx = [i for i, x in enumerate(ds.meta) if x["method"] == "stegastamp"][:6]
x = torch.stack([torch.from_numpy(ds[i]["image"]).permute(2, 0, 1) for i in idx]).float().div(255)
xin = wm._prep(x).astype(np.float32)
outs = sess.run(extra, {"image": xin, "secret": np.zeros((len(xin), 100), np.float32)})
s = outs[0]
print("decoder conv input:", s.shape, "image input:", xin.shape)
if s.shape == xin.shape:
    print("  MSE(stn_out, input) per image:", np.round(((s - xin) ** 2).mean((1, 2, 3)), 6).tolist())
    print("  corr(stn_out, input) per image:", [round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 5) for a, b in zip(s, xin)])
for t, o in zip(mm, outs[1:]):
    o = np.asarray(o)
    print(f"  {t}: shape {o.shape}" + (f" values img0 {np.round(o[0].ravel()[:8], 4).tolist()}" if o.ndim == 2 and o.shape[1] <= 8 else ""))
print("\nD1 DONE")
