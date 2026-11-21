"""E04 step 0: StegaStamp STN geometry. Facts only, no training.
1. find the affine parameters (theta) feeding the gen_stn sampler; expose them and the decoder-conv input;
2. per train image: displacement implied by theta (pixels at 400) relative to identity;
3. numeric round-trip check: warp the STN output back to the input frame with the inverse affine (torch
   grid_sample, both align_corners conventions, both theta interpretations) and compare to the input image.
   The convention that maximises the correlation is the real one (must reach ~1 inside the valid region)."""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
P = R.P
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

src = P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx"
model = onnx.load(str(src)); g = model.graph
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)
# theta = consumer chain of gen_stn/MatMul:0 (expected MatMul -> Add(identity bias) -> Reshape [N,2,3])
chain, t = [], "gen_stn/MatMul:0"
for _ in range(4):
    nx = cons.get(t, [])
    if not nx:
        break
    chain.append((nx[0].op_type, nx[0].output[0], list(nx[0].input))); t = nx[0].output[0]
print("theta chain:", chain)
inits = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
for op, out, ins in chain:
    for i in ins:
        if i in inits and inits[i].size <= 12:
            print("  constant", i, inits[i].ravel().tolist())
theta_name = next((out for op, out, ins in chain if op in ("Add", "BiasAdd")), chain[0][1])
expose = [theta_name, "gen_decoder/sub:0"]
for t in expose:
    vi = onnx.helper.make_empty_tensor_value_info(t); vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    model.graph.output.append(vi)
out_dir = Path("/p/project1/training2666/jacquet2/research/E04"); out_dir.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(out_dir / "stega_theta.onnx"))
sess = ort.InferenceSession(str(out_dir / "stega_theta.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# what does gen_decoder/sub subtract?
sub = next(n for n in g.node if n.output[0] == "gen_decoder/sub:0")
print("gen_decoder/sub inputs:", list(sub.input), [inits[i].ravel()[:4].tolist() for i in sub.input if i in inits])

ds = LocalisationDataset(P.TASK_ROOT / "release", "train")
wm = load_watermark("stegastamp", P.TASK_ROOT / "models", "cuda")
idx = [i for i, m in enumerate(ds.meta) if m["method"] == "stegastamp"]
thetas, best = [], {}
for s in range(0, len(idx), 34):
    x = torch.stack([torch.from_numpy(ds[i]["image"]).permute(2, 0, 1) for i in idx[s:s + 34]]).float().div(255)
    xin = wm._prep(x).astype(np.float32)
    th, so = sess.run(expose, {"image": xin, "secret": np.zeros((len(xin), 100), np.float32)})
    thetas.append(np.asarray(th).reshape(len(xin), -1))
    if s == 0:                                                      # round-trip on the first batch
        inp = torch.from_numpy(xin).permute(0, 3, 1, 2)             # N,3,400,400
        so_t = torch.from_numpy(np.asarray(so)).permute(0, 3, 1, 2)
        # undo the subtraction constant if it is a scalar initializer
        cst = [inits[i] for i in sub.input if i in inits]
        if cst and cst[0].size == 1:
            so_t = so_t + float(cst[0])
        A = torch.from_numpy(thetas[0][:, :6]).float().view(-1, 2, 3)
        print("theta img0:", A[0].tolist())
        print("corr(stn_out, input) raw:", [round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 4) for a, b in zip(so_t[:6].numpy(), inp[:6].numpy())])
        for interp in ("theta_maps_out_to_in", "theta_maps_in_to_out"):
            for ac in (False, True):
                M = torch.cat([A, torch.tensor([[[0, 0, 1.0]]]).expand(len(A), 1, 3)], 1)
                Minv = torch.linalg.inv(M)[:, :2]
                # grid_sample(img, grid) reads img at grid(p): to bring stn_out back to the input frame we read
                # stn_out at out-coords(p_in) = T^-1(p_in) if theta maps out->in.
                warp = Minv if interp == "theta_maps_out_to_in" else A
                grid = F.affine_grid(warp, list(inp.shape), align_corners=ac)
                back = F.grid_sample(so_t, grid, align_corners=ac, padding_mode="zeros")
                valid = F.grid_sample(torch.ones_like(so_t[:, :1]), grid, align_corners=ac) > 0.99
                cs = []
                for b_, i_, v_ in zip(back, inp, valid):
                    v = v_[0]
                    cs.append(float(np.corrcoef(b_[:, v].ravel(), i_[:, v].ravel())[0, 1]))
                best[(interp, ac)] = np.mean(cs)
                print(f"round-trip {interp} align_corners={ac}: corr {np.round(cs[:6], 4).tolist()} mean {np.mean(cs):.4f} "
                      f"valid frac {float(valid.float().mean()):.3f}")
th = np.concatenate(thetas)[:, :6].reshape(-1, 2, 3)
ident = np.array([[1, 0, 0], [0, 1, 0]], float)
# displacement in pixels at 400 for corners/centre of the normalised square
pts = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1], [0, 0, 1]], float).T
disp = np.abs(np.einsum("nij,jk->nik", th - ident, pts)) * 200
md = disp.max(axis=(1, 2)); cen = disp[:, :, 4].max(1)
print(f"theta - identity: mean |.| {np.abs(th - ident).mean():.4f}; max displacement px@400: median {np.median(md):.1f} "
      f"p90 {np.percentile(md, 90):.1f} max {md.max():.1f}; centre shift median {np.median(cen):.1f}")
print("best round-trip convention:", max(best, key=best.get), round(max(best.values()), 4))
np.save(out_dir / "theta_train.npy", th)
print("E04 INSPECT DONE")
