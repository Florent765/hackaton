"""E04: extract the StegaStamp STN affine parameters for every image (train + test) with a private ONNX copy.
Writes <research_cache>/E04/{split}_stegastamp_theta.pt = {"names", "theta": [N,2,3] float32, "onnx_sha"}.
Convention verified by inspect_geometry.py: theta maps STN-output coords -> input coords (normalised, the
round-trip with affine_grid(inverse) reaches corr 0.983 vs 0.57 for the opposite convention)."""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

OUT = Path("/p/project1/training2666/jacquet2/research_cache/E04"); OUT.mkdir(parents=True, exist_ok=True)
src = P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx"
sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
m = onnx.load(str(src))
vi = onnx.helper.make_empty_tensor_value_info("gen_stn/add:0"); vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
m.graph.output.append(vi)
onnx.save(m, str(OUT / "stega_theta.onnx"))
sess = ort.InferenceSession(str(OUT / "stega_theta.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cpu")
for split in ("train", "test"):
    out = OUT / f"{split}_stegastamp_theta.pt"
    if out.exists():
        print("exists", out); continue
    ds = LocalisationDataset(P.TASK_ROOT / "release", split); pos = {x["image_name"]: i for i, x in enumerate(ds.meta)}
    names = P.load_cache("stegastamp", split)["names"]; th = []; t0 = time.time()
    for s in range(0, len(names), 32):
        x = torch.stack([torch.from_numpy(ds[pos[n]]["image"]).permute(2, 0, 1) for n in names[s:s + 32]]).float().div(255)
        th.append(np.asarray(sess.run(["gen_stn/add:0"], {"image": wm._prep(x).astype(np.float32),
                                                          "secret": np.zeros((len(x), 100), np.float32)})[0]).reshape(-1, 2, 3))
    th = torch.from_numpy(np.concatenate(th)).float()
    tmp = out.with_suffix(f".tmp{os.getpid()}"); torch.save({"names": names, "theta": th, "onnx_sha": sha}, tmp); os.replace(tmp, out)
    print(f"{split}: {len(names)} thetas in {time.time() - t0:.0f}s, |theta-I| mean {float((th - torch.tensor([[1., 0, 0], [0, 1, 0]])).abs().mean()):.4f}")
print("---\nval_miou: 0.000000\nE04 THETA DONE")
