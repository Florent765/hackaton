"""A02 step 0 (analysis only, labels used for diagnostics, nothing fitted): how rectangular are the train masks?
Per method x regime (hi = bit accuracy > 0.75 as in rlib; full_watermark only for reporting):
  fill = area / bbox area of the positive class (lo regime) or of the negative class (hi regime);
  mIoU of the oracle 'bbox' mask vs the GT; mIoU of N03_attr OOF (seed 0, confirm thresholds) for reference."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402


def bbox_fill(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.nan, m
    b = np.zeros_like(m); b[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return m.sum() / b.sum(), b


rec = torch.load(R.OUT_ROOT / "N03_attr" / "seed0.pt", weights_only=False)
S = R.summarize([rec])
for method, mr in rec["methods"].items():
    masks = mr["masks"].numpy().astype(np.uint8); hi = mr["hi"].numpy(); fw = mr["full_watermark"].numpy()
    conf = S["per_image"][method]["confirm"].numpy()
    for reg, sel in (("lo", ~hi), ("hi", hi)):
        fills, ob = [], []
        for i in np.nonzero(sel)[0]:
            m = masks[i] if reg == "lo" else 1 - masks[i]
            f, b = bbox_fill(m)
            fills.append(f)
            pred = b if reg == "lo" else 1 - b
            ob.append(float(R.miou_batch(torch.from_numpy(pred[None]), torch.from_numpy(masks[i][None]))[0]))
        fills = np.array(fills)
        print(f"{method} {reg}: n={sel.sum()} fw_true={fw[sel].mean():.2f} fill median {np.nanmedian(fills):.3f} "
              f"frac>0.9 {np.nanmean(fills > 0.9):.2f} frac>0.98 {np.nanmean(fills > 0.98):.2f} | bbox-oracle mIoU {np.mean(ob):.4f} "
              f"| N03 confirm {conf[sel].mean():.4f} cov mean {masks[sel].mean():.3f}")
print("---\nval_miou: 0.000000")
