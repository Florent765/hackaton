"""A10 decoding rule (no retraining): replace the N01 mask by the decoding-refined rectangle when the verifier is
confident. For PixelSeal / TrustMark images, best refined box b* (prepare_win.py) with score z* = max rz:
    mask = box(b*) if (lo flag: bit acc <= 0.75) and z* > tau else N01 mask
tau per method and outer fold is chosen on the outer-training images only, with their INNER predictions (the same
inner-fold calibration eval_n01 uses), so the evaluated fold's labels are never used (confirmatory). Grid includes
tau = inf (= plain N01). Also prints a diagnostic table (box IoU vs N01 IoU by z* bin) and the random-message control
(rule driven by the control boxes / scores).
usage: python eval_box.py <exp> [seeds...]"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
from compare import boot_ci  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


N01 = _load(HERE.parent / "N01" / "eval_n01.py", "eval_n01")
W = _load(HERE / "exp.py", "a10_exp")
DEV = "cuda"
TAUS = [float("inf"), 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]


def box_masks(rbox, rz):
    """best refined box per image -> [N,128,128] bool (4 px per cell)"""
    b = rbox[torch.arange(len(rz)), rz.argmax(1)].float() / 4
    e = torch.arange(128, dtype=torch.float32) + 0.5
    my = (e[None] >= b[:, 0:1]) & (e[None] < b[:, 2:3]); mx = (e[None] >= b[:, 1:2]) & (e[None] < b[:, 3:4])
    return (my[:, :, None] & mx[:, None, :])


def apply(n01m, boxm, zs, lo, tau):
    use = lo & (zs > tau)
    return torch.where(use[:, None, None], boxm, n01m), use


def main():
    exp = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or None
    d = R.OUT_ROOT / exp
    recs = [torch.load(p, weights_only=False) for p in sorted(d.glob("seed*.pt")) if seeds is None or int(p.stem[4:]) in seeds]
    print(f"{exp}: seeds {[r['seed'] for r in recs]}")
    allv = {k: [] for k in ("n01", "box", "box_ctrl")}; meth = []
    for method in recs[0]["methods"]:
        ms = [r["methods"][method] for r in recs]
        p = torch.stack([m["oof"].float() for m in ms]).mean(0).to(DEV)
        gt = ms[0]["masks"].float().to(DEV); fold = ms[0]["fold"].to(DEV); lo = ~ms[0]["hi"].to(DEV)
        conf = {k: torch.zeros(len(gt), device=DEV) for k in allv}
        if method in ("pixelseal", "trustmark"):
            o = W.load_win("train", method)
            assert list(o["names"]) == list(ms[0]["names"]), "A10 cache not aligned"
            boxes = {s: (box_masks(o["rbox" + s], o["rz" + s].float()).to(DEV), o["rz" + s].float().max(1).values.to(DEV))
                     for s in ("", "_ctrl")}
            # diagnostic (all OOF, exploratory only): IoU of box vs N01 mask by z* bin, lo images
            tb = N01.best_tb(p, gt); nm = N01.kstar_masks(N01.calib(p, *tb))
            ib = R.miou_batch(boxes[""][0], gt); inn = R.miou_batch(nm, gt); zs = boxes[""][1]
            for lo_, hi_ in ((-9, 4), (4, 6), (6, 8), (8, 10), (10, 99)):
                sel = lo & (zs >= lo_) & (zs < hi_)
                if sel.any():
                    print(f"  diag {method} lo z*[{lo_},{hi_}) n={int(sel.sum())} box mIoU {float(ib[sel].mean()):.3f} "
                          f"N01 mIoU {float(inn[sel].mean()):.3f}")
        else:
            boxes = None
        for k in range(R.P.K):
            idx = ms[0]["inner"][k]["idx"].to(DEV); va = (fold == k).nonzero()[:, 0]
            ip = torch.stack([m["inner"][k]["probs"].float() for m in ms]).mean(0).to(DEV)
            t = N01.best_tb(ip, gt[idx])
            n_tr = N01.kstar_masks(N01.calib(ip, *t)); n_va = N01.kstar_masks(N01.calib(p[va], *t))
            conf["n01"][va] = R.miou_batch(n_va, gt[va])
            for key, suf in (("box", ""), ("box_ctrl", "_ctrl")):
                if boxes is None:
                    conf[key][va] = conf["n01"][va]; continue
                bm, zs = boxes[suf]
                sc = {tau: float(R.miou_batch(apply(n_tr, bm[idx], zs[idx], lo[idx], tau)[0], gt[idx]).mean()) for tau in TAUS}
                tau = max(sc, key=sc.get)
                mk, use = apply(n_va, bm[va], zs[va], lo[va], tau)
                conf[key][va] = R.miou_batch(mk, gt[va])
                print(f"  {method} fold {k} {key}: tau {tau} (train {sc[tau]:.4f} vs inf {sc[float('inf')]:.4f}), "
                      f"applied to {int(use.sum())}/{len(va)}")
        for key in allv:
            allv[key] += conf[key].tolist()
        meth += [method] * len(gt)
        print(f"{method}: confirm n01 {float(conf['n01'].mean()):.4f} box {float(conf['box'].mean()):.4f} "
              f"box_ctrl {float(conf['box_ctrl'].mean()):.4f}")
    meth = np.array(meth); base = np.array(allv["n01"])
    for key in ("box", "box_ctrl"):
        a = np.array(allv[key]); dl = a - base; lo_, hi_ = boot_ci(dl, meth)
        print(f"confirm {key}: n01 {base.mean():.4f} -> {a.mean():.4f} delta {dl.mean():+.4f} CI95 [{lo_:+.4f},{hi_:+.4f}] "
              f"W/L {(dl > 1e-9).sum()}/{(dl < -1e-9).sum()}")
    print("---\nval_miou: 0.000000\nBOX DONE")


if __name__ == "__main__":
    main()
