"""Generic final submission: any registered experiment (run_exp registry) + N01 expected-mIoU decoding.
For each method: 3 heads (seeds 0,1,2) trained on ALL train images with that experiment's cfg/inputs, test
probabilities averaged; N01 calibration (T, b) per method chosen on the experiment's 3-seed OOF probabilities.
Validates and writes the .pt. usage: python predict_final.py <exp_id> <out_path.pt>"""
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "N01"))
import rlib as R  # noqa: E402
import run_exp  # noqa: E402
from eval_n01 import best_tb, calib, kstar_masks  # noqa: E402

exp, out_path = sys.argv[1], Path(sys.argv[2]); P = R.P
cfg, extra, head, notes = run_exp.resolve(exp); head = head or R.default_head
oof = {}
for s in (0, 1, 2):
    rec = torch.load(R.OUT_ROOT / exp / f"seed{s}.pt", weights_only=False)
    for m, v in rec["methods"].items():
        oof.setdefault(m, []).append(v["oof"].float())
names_all, masks_all, t0 = [], [], time.time()
for method in P.METHODS:
    c = P.load_cache(method); f = R.build_inputs(c, "train", cfg, extra); y = c["masks"].to(R.DEV).float()
    T, b = best_tb(torch.stack(oof[method]).mean(0).to(R.DEV), y)
    ct = P.load_cache(method, "test"); ft = R.build_inputs(ct, "test", cfg, extra)
    assert set(ft) == set(f) and all(ft[k].shape[1:] == f[k].shape[1:] for k in f), "train/test inputs differ"
    pt = sum(R.predict(R.fit(f, y, list(range(len(y))), s, cfg, head), ft, list(range(len(ct["names"])))) for s in (0, 1, 2)) / 3
    mk = kstar_masks(calib(pt, T, b)).cpu()
    torch.save({"names": ct["names"], "probs": pt.half().cpu(), "T": T, "b": b}, out_path.with_name(out_path.stem + f"_{method}_probs.pt"))
    names_all += ct["names"]; masks_all.append(mk)
    print(f"{method}: N01 T={T} b={b}; {len(ct['names'])} test masks, coverage {float(mk.float().mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)
size = P.write_submission(names_all, torch.cat(masks_all), out_path)
print(f"submission {out_path} {size:.2f} MB validated ({exp} + N01: {notes})")
print("---\nval_miou: 0.000000\nFINAL DONE")
