"""AR_n03_dense: StegaStamp exact attribution (N03) densified by shift-and-stitch of the 400-px decoder input.
dec_conv_7 has stride ~400/13; the decoder input is circularly shifted by (dy, dx) = (a, b) * stride/4 for
a, b in 0..3 and the 13x13 maps interleaved into 52x52. Same exact decomposition per pass (gates per pass).
Writes <research_cache>/AR/{split}_stegastamp_attr_dense.pt {"names", "attr": [N,1,52,52]}."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
import onnx  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/AR"); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("/p/project1/training2666/jacquet2/research_cache/N03/stega_attr.onnx")   # copy built by N03
m = onnx.load(str(P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx")); g = m.graph
init = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}; byout = {n.output[0]: n for n in g.node}
mm1 = byout["gen_decoder/dec_dense/MatMul:0"]; mm2 = byout["gen_decoder/decoder_out/MatMul:0"]
h_name = [i for i in mm1.input if i not in init][0]
W1 = torch.from_numpy(init[[i for i in mm1.input if i in init][0]]).float().cuda()
W2 = torch.from_numpy(init[[i for i in mm2.input if i in init][0]]).float().cuda()
sess = ort.InferenceSession(str(SRC), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
names_out = [h_name, "gen_decoder/dec_dense/BiasAdd:0", "gen_decoder/dec_conv_7/Relu:0"]
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cpu")
S = 4
for split in ("train", "test"):
    out = OUT / f"{split}_stegastamp_attr_dense.pt"
    if out.exists():
        print("exists", out); continue
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    names = P.load_cache("stegastamp", split)["names"]; res = []; t0 = time.time()
    for s in range(0, len(names), 32):
        items = [ds[pos[n]] for n in names[s:s + 32]]
        x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div(255)
        xin = wm._prep(x).astype(np.float32)                                     # N,400,400,3
        sg = 2 * torch.from_numpy(np.stack([it["message"] for it in items])).float().cuda() - 1
        dense = None
        for a in range(S):
            for b in range(S):
                dy, dx = int(round(a * 400 / 13 / S)), int(round(b * 400 / 13 / S))
                xs = np.roll(xin, shift=(-dy, -dx), axis=(1, 2))
                h, z1, c7 = sess.run(names_out, {"image": xs, "secret": np.zeros((len(xs), 100), np.float32)})
                h = torch.from_numpy(np.asarray(h)).float().cuda(); z1 = torch.from_numpy(np.asarray(z1)).float().cuda()
                C, H, W = np.asarray(c7).shape[1:]
                u = ((sg @ W2.T) * (z1 > 0).float()) @ W1.T
                mp = (h * u).view(-1, H, W, C).sum(-1)
                if dense is None:
                    dense = torch.zeros(len(xs), H * S, W * S)
                dense[:, a::S, b::S] = mp.cpu()
        res.append(dense[:, None])
    rec = {"names": names, "attr": torch.cat(res)}
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save(rec, tmp); os.replace(tmp, out)
    print(f"{split}: {tuple(rec['attr'].shape)} in {time.time() - t0:.0f}s", flush=True)
print("---\nval_miou: 0.000000\nAR DENSE DONE")
