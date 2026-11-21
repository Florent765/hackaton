"""Final submission for the best confirmed configuration: N03 (REF + StegaStamp exact attribution) + N01
(expected-mIoU top-k decoding). For each method: 3 heads (seeds 0,1,2) trained on ALL train images with the
research protocol (rlib), test probabilities averaged; N01 calibration (T, b) per method chosen on the 3-seed OOF
probabilities of N03_attr (CV estimate 0.5244 exploratory / 0.5231 confirmatory). Writes and validates the .pt.
usage: python predict_n03_n01.py <out_path.pt>"""
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "N01"))
import rlib as R  # noqa: E402
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("n03exp", HERE.parent / "N03" / "exp.py"); N03 = importlib.util.module_from_spec(_s); _s.loader.exec_module(N03)
from eval_n01 import best_tb, calib, kstar_masks  # noqa: E402

out_path = Path(sys.argv[1]); P = R.P
cfg = dict(R.REF); extra = N03.make_extra("attr")
oof = {}
for s in (0, 1, 2):
    rec = torch.load(R.OUT_ROOT / "N03_attr" / f"seed{s}.pt", weights_only=False)
    for m, v in rec["methods"].items():
        oof.setdefault(m, []).append(v["oof"].float())
names_all, masks_all, t0 = [], [], time.time()
for method in P.METHODS:
    c = P.load_cache(method); f = R.build_inputs(c, "train", cfg, extra); y = c["masks"].to(R.DEV).float()
    p_oof = torch.stack(oof[method]).mean(0).to(R.DEV)
    T, b = best_tb(p_oof, y)
    ct = P.load_cache(method, "test"); ft = R.build_inputs(ct, "test", cfg, extra)
    assert set(ft) == set(f) and all(ft[k].shape[1:] == f[k].shape[1:] for k in f), "train/test inputs differ"
    idx_all = list(range(len(y))); idx_te = list(range(len(ct["names"])))
    pt = sum(R.predict(R.fit(f, y, idx_all, s, cfg), ft, idx_te) for s in (0, 1, 2)) / 3
    mk = kstar_masks(calib(pt, T, b)).cpu()
    torch.save({"names": ct["names"], "probs": pt.half().cpu(), "T": T, "b": b}, out_path.with_name(out_path.stem + f"_{method}_probs.pt"))
    names_all += ct["names"]; masks_all.append(mk)
    print(f"{method}: N01 calib T={T} b={b}; test {len(idx_te)} masks, mean predicted coverage {float(mk.float().mean()):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
size = P.write_submission(names_all, torch.cat(masks_all), out_path)
print(f"submission {out_path} {size:.2f} MB (validated: keys, 1020 unique names, [1020,128,128] bool, <= 20 MB)")
print("---\nval_miou: 0.000000\nFINAL DONE")
