"""N03: exact per-position message attribution for StegaStamp (agent idea N03).
Decoder tail: h = flatten(dec_conv_7) -> z1 = h W1 + b1 -> a = relu(z1) -> logits = a W2 + b2.
With the ReLU gates g = 1[z1 > 0] fixed, sum_k s_k logit_k = sum_i h_i u_i + const, u = W1 (g * (W2 s)),
s = 2m - 1. The map h_i u_i summed over channels (positions in the STN-output frame) is an exact decomposition of
the image-level agreement, i.e. the StegaStamp analogue of per_position_logits (which PixelSeal/TrustMark expose).
Checks: z1 reconstructed from exposed h matches the graph; sum of the map + const = signed logit sum.
Writes <research_cache>/N03/{split}_stegastamp_attr.pt {"names","attr":[N,1,H,W],"ctrl":[N,1,H,W]} (ctrl = random
message) and prints tie-aware per-image AUROC on train."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from scipy.stats import rankdata  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/N03"); OUT.mkdir(parents=True, exist_ok=True)
m = onnx.load(str(P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx")); g = m.graph
init = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
byout = {n.output[0]: n for n in g.node}
mm1 = byout["gen_decoder/dec_dense/MatMul:0"]; add1 = byout["gen_decoder/dec_dense/BiasAdd:0"]
mm2 = byout["gen_decoder/decoder_out/MatMul:0"]; add2 = byout["gen_decoder/decoder_out/BiasAdd:0"]
h_name = [i for i in mm1.input if i not in init][0]
W1 = init[[i for i in mm1.input if i in init][0]]; b1 = init[[i for i in add1.input if i in init][0]]
W2 = init[[i for i in mm2.input if i in init][0]]; b2 = init[[i for i in add2.input if i in init][0]]
print("h:", h_name, "W1", W1.shape, "b1", b1.shape, "W2", W2.shape, "b2", b2.shape)
# shape of conv7 output (NHWC before flatten): find the transpose feeding the flatten chain
conv7 = "gen_decoder/dec_conv_7/Relu:0"
expose = [h_name, "gen_decoder/dec_dense/BiasAdd:0", conv7]
for t in expose:
    vi = onnx.helper.make_empty_tensor_value_info(t); vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    m.graph.output.append(vi)
onnx.save(m, str(OUT / "stega_attr.onnx"))
sess = ort.InferenceSession(str(OUT / "stega_attr.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
W1t, b1t, W2t, b2t = (torch.from_numpy(np.asarray(a, np.float32)).cuda() for a in (W1, b1, W2, b2))
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cpu")


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


checked = False
for split in ("train", "test"):
    out = OUT / f"{split}_stegastamp_attr.pt"
    if out.exists():
        print("exists", out); continue
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    names = P.load_cache("stegastamp", split)["names"]; rng = np.random.RandomState(7)
    A, C = [], []; t0 = time.time()
    for s in range(0, len(names), 32):
        items = [ds[pos[n]] for n in names[s:s + 32]]
        x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255)
        h, z1, c7 = sess.run(expose, {"image": wm._prep(x).astype(np.float32), "secret": np.zeros((len(x), 100), np.float32)})
        h = torch.from_numpy(np.asarray(h)).float().cuda(); z1 = torch.from_numpy(np.asarray(z1)).float().cuda()
        c7 = np.asarray(c7)
        if not checked:
            print("conv7 output shape (graph layout):", c7.shape, "h", tuple(h.shape))
            print("max |h W1 + b1 - z1| =", float((h @ W1t + b1t - z1).abs().max()))
        gate = (z1 > 0).float()
        sg = 2 * torch.from_numpy(np.stack([it["message"] for it in items])).float().cuda() - 1
        ctrl = 2 * torch.from_numpy(rng.randint(0, 2, sg.shape)).float().cuda() - 1
        maps = []
        for sv in (sg, ctrl):
            u = ((sv @ W2t.T) * gate) @ W1t.T                              # [B, D]
            maps.append(h * u)
        if not checked:
            logits = torch.relu(z1) @ W2t + b2t
            lhs = (sg * logits).sum(1); rhs = maps[0].sum(1) + (sg * ((gate * b1t) @ W2t + b2t)).sum(1)
            print("linearity check max |sum_k s_k logit_k - (sum_i map_i + const)| =", float((lhs - rhs).abs().max()),
                  " (|lhs| mean", float(lhs.abs().mean()), ")")
            checked = True
        # flatten order: TF flattens NHWC; conv7 exposed by ONNX is NCHW after the transpose of the conv -> derive H,W,C
        D = h.shape[1]
        Cch, H, W = c7.shape[1:]                    # Relu output is NCHW; a Transpose to NHWC precedes the flatten
        assert Cch * H * W == D, (c7.shape, D)
        if s == 0 and split == "train":            # verify the NHWC flatten order numerically
            c7t = torch.from_numpy(c7).float().cuda().permute(0, 2, 3, 1).reshape(len(c7), -1)
            print("max |flatten_NHWC(conv7) - h| =", float((c7t - h).abs().max()))
        for k, mp in enumerate(maps):
            (A if k == 0 else C).append(mp.view(-1, H, W, Cch).sum(-1)[:, None].cpu())   # NHWC flatten order
    rec = {"names": names, "attr": torch.cat(A), "ctrl": torch.cat(C), "layout": f"H{H}W{W}C{Cch} NHWC flatten"}
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{split}: {len(names)} maps {tuple(rec['attr'].shape)} in {time.time() - t0:.0f}s", flush=True)
    if split == "train":
        masks = P.load_cache("stegastamp")["masks"].numpy()
        for key in ("attr", "ctrl"):
            up = F.interpolate(rec[key].float(), size=128, mode="bilinear", align_corners=False)[:, 0].numpy()
            print(f"  AUROC {key}: {np.nanmean([auroc(up[i], masks[i]) for i in range(len(masks))]):.4f}", flush=True)
print("---\nval_miou: 0.000000\nN03 ATTR DONE")
