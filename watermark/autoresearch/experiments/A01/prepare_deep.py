"""A01: exact message attribution for StegaStamp at deeper (finer) decoder layers + shift-TTA of the N03 map.
N03 decomposes sum_k s_k logit_k over the 13x13 positions of dec_conv_7 (gates of dec_dense fixed). Here the same
objective obj = <flatten_NHWC(c7), u>, u = W1 (g * (W2 s)), is back-propagated through dec_conv_7/6/5 (ReLU gates
fixed => piecewise linear, grad x input is an exact decomposition up to the conv bias terms) to get maps at
c6 (25x25), c5 (50x50) and c4 (100x100). The conv tail is re-implemented in torch from the ONNX initialisers
(pads from the graph) and checked against the ONNX activations.
tta7: N03 map averaged over 4x4 input shifts of 8 px (400-px frame, reflect padding), each mapped back.
Writes research_cache/A01/{split}_stegastamp_deep.pt {names, a7,a6,a5,a4,tta7 and *_ctrl (random message)}."""
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

OUT = Path("/p/project1/training2666/jacquet2/research_cache/A01"); OUT.mkdir(parents=True, exist_ok=True)
m = onnx.load(str(P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx")); g = m.graph
init = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
byout = {n.output[0]: n for n in g.node}


def T(a):
    return torch.from_numpy(np.array(a, np.float32)).cuda()


def dense(name):
    mm = byout[f"gen_decoder/{name}/MatMul:0"]; ad = byout[f"gen_decoder/{name}/BiasAdd:0"]
    return T(init[[i for i in mm.input if i in init][0]]), T(init[[i for i in ad.input if i in init][0]])


W1, b1 = dense("dec_dense"); W2, b2 = dense("decoder_out")
CONV = {}
for k in (5, 6, 7):
    n = byout[f"gen_decoder/dec_conv_{k}/BiasAdd:0"]
    at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    pt, pl, pb, pr = at["pads"]
    CONV[k] = (T(init[n.input[1]]), T(init[n.input[2]]), (pl, pr, pt, pb), at["strides"][0])
    print(k, tuple(CONV[k][0].shape), CONV[k][2], CONV[k][3])


def conv(x, k):
    w, b, pad, s = CONV[k]
    return torch.relu(F.conv2d(F.pad(x, pad), w, b, stride=s))


names_out = ["gen_decoder/dec_conv_4/Relu:0", "gen_decoder/dec_conv_5/Relu:0", "gen_decoder/dec_conv_6/Relu:0",
             "gen_decoder/dec_conv_7/Relu:0", "gen_decoder/dec_dense/BiasAdd:0"]
for t in names_out:
    vi = onnx.helper.make_empty_tensor_value_info(t); vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    m.graph.output.append(vi)
onnx.save(m, str(OUT / "stega_deep.onnx"))
sess = ort.InferenceSession(str(OUT / "stega_deep.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cpu")


def auroc(score, label):
    s = np.asarray(score, float).ravel(); y = np.asarray(label).ravel().astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def run(xn):
    return [torch.from_numpy(np.asarray(o)).float().cuda() for o in
            sess.run(names_out, {"image": xn.astype(np.float32), "secret": np.zeros((len(xn), 100), np.float32)})]


def maps_for(c4, z1, svs, check=False):
    gate = (z1 > 0).float(); res = []
    for sv in svs:
        u = ((sv @ W2.T) * gate) @ W1.T
        x4 = c4.clone().requires_grad_(True)
        x5 = conv(x4, 5); x5.retain_grad(); x6 = conv(x5, 6); x6.retain_grad(); x7 = conv(x6, 7); x7.retain_grad()
        h = x7.permute(0, 2, 3, 1).reshape(len(x7), -1)
        (h * u).sum().backward()
        res.append({f"a{k}": (x * x.grad).sum(1, keepdim=True).detach() for k, x in ((4, x4), (5, x5), (6, x6), (7, x7))})
        if check:
            res[-1]["_x"] = (x5.detach(), x6.detach(), x7.detach())
    return res


SH = [(dy, dx) for dy in (0, 8, 16, 24) for dx in (0, 8, 16, 24)]
checked = False
for split in ("train", "test"):
    out = OUT / f"{split}_stegastamp_deep.pt"
    if out.exists():
        print("exists", out); continue
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    names = P.load_cache("stegastamp", split)["names"]; rng = np.random.RandomState(7)
    acc = {}; t0 = time.time()
    for s in range(0, len(names), 32):
        items = [ds[pos[n]] for n in names[s:s + 32]]
        x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255)
        xn = wm._prep(x)                                                     # [B,400,400,3]
        sg = 2 * torch.from_numpy(np.stack([it["message"] for it in items])).float().cuda() - 1
        ctrl = 2 * torch.from_numpy(rng.randint(0, 2, sg.shape)).float().cuda() - 1
        c4, c5, c6, c7, z1 = run(xn)
        r = maps_for(c4, z1, (sg, ctrl), check=not checked)
        if not checked:
            x5, x6, x7 = r[0].pop("_x"); r[1].pop("_x")
            print("max|c5-onnx|", float((x5 - c5).abs().max()), "max|c6-onnx|", float((x6 - c6).abs().max()),
                  "max|c7-onnx|", float((x7 - c7).abs().max()), "c7 max", float(c7.abs().max()))
            print("sum a7 / a6 / a5 (first 3):", r[0]["a7"].sum((1, 2, 3))[:3].tolist(), r[0]["a6"].sum((1, 2, 3))[:3].tolist(),
                  r[0]["a5"].sum((1, 2, 3))[:3].tolist())
            checked = True
        # shift TTA of the conv7 map: content shifted by -d, map upsampled to 400 and shifted back by +d
        tt = [torch.zeros(len(x), 1, 400, 400, device="cuda") for _ in range(2)]
        xt = torch.from_numpy(xn).permute(0, 3, 1, 2)
        for dy, dx in SH:
            xs = F.pad(xt, (0, dx, 0, dy), mode="reflect")[:, :, dy:dy + 400, dx:dx + 400]
            c4s, _, _, _, zs = run(xs.permute(0, 2, 3, 1).contiguous().numpy())
            rs = maps_for(c4s, zs, (sg, ctrl))
            for j in range(2):
                up = F.interpolate(rs[j]["a7"], size=400, mode="bilinear", align_corners=False)
                up = F.pad(up, (dx, 0, dy, 0), mode="replicate")[:, :, :400, :400]
                tt[j] += up / len(SH)
        for j, suf in ((0, ""), (1, "_ctrl")):
            for k, v in r[j].items():
                acc.setdefault(k + suf, []).append(v.cpu())
            acc.setdefault("tta7" + suf, []).append(F.adaptive_avg_pool2d(tt[j], 50).cpu())
    rec = {"names": names, **{k: torch.cat(v) for k, v in acc.items()}}
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{split}: {len(names)} in {time.time() - t0:.0f}s", {k: tuple(v.shape) for k, v in rec.items() if k != "names"}, flush=True)
    if split == "train":
        masks = P.load_cache("stegastamp")["masks"].numpy()
        n03 = torch.load("/p/project1/training2666/jacquet2/research_cache/N03/train_stegastamp_attr.pt", weights_only=False)
        print("  corr a7 vs N03 attr:", float(np.corrcoef(rec["a7"].flatten().numpy(), n03["attr"].flatten().numpy())[0, 1]))
        for key in ("a7", "a6", "a5", "a4", "tta7", "a7_ctrl", "a6_ctrl", "a5_ctrl", "a4_ctrl", "tta7_ctrl"):
            up = F.interpolate(rec[key].float(), size=128, mode="bilinear", align_corners=False)[:, 0].numpy()
            print(f"  AUROC {key}: {np.nanmean([auroc(up[i], masks[i]) for i in range(len(masks))]):.4f}", flush=True)
print("---\nval_miou: 0.000000\nA01 DEEP DONE")
