"""Entry point for research runs: python run_exp.py --exp <ID> --seed <s>
Each ID maps to an explicit configuration (all derived from rlib.REF, one conceptual change each)."""
import argparse
import copy
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))
import rlib as R  # noqa: E402


def cfg(**kw):
    c = copy.deepcopy(R.REF); c.update(kw); return c


def mod(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), HERE / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# id -> (cfg, extra_inputs factory or None, make_head or None, notes)
REGISTRY = {
    "REF": (lambda: (cfg(), None, None, "reference e29 protocol (current transform, del8 sidecars)")),
    "E01_signed": (lambda: (cfg(transform="signed"), None, None, "sign(x)*log1p(|x|) instead of log1p(clamp(x,0))")),
    "E01_identity": (lambda: (cfg(transform="identity"), None, None, "raw maps, no transform")),
}


def resolve(exp):
    if exp in REGISTRY:
        return REGISTRY[exp]()
    # experiment modules may register more ids: experiments/E##/exp.py with REGISTRY = {id: fn}
    for p in sorted(list(HERE.glob("E*/exp.py")) + list(HERE.glob("N*/exp.py")) + list(HERE.glob("A*/exp.py"))):
        m = mod(str(p.relative_to(HERE)).replace("\\", "/"))
        if exp in getattr(m, "REGISTRY", {}):
            return m.REGISTRY[exp](cfg)
    raise KeyError(exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c, extra, head, notes = resolve(a.exp)
    R.run(a.exp, c, a.seed, extra_inputs=extra, make_head=head or R.default_head, notes=notes)
