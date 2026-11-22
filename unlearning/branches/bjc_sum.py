"""Compact summary of DIAG/CTRL lines in a run.log (login-node helper). usage: python3 bjc_sum.py run.log [skip]"""
import json, sys
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, l in enumerate(open(sys.argv[1])):
    if l.startswith(("DIAG", "CTRL")):
        _, n, j = l.split(" ", 2); d = json.loads(j)
        if n.endswith("_TEMPCTRL"):
            print(f"   ctrl s={d['scale']} cm kl {d['cm']['kl']} mild kl {d['mild']['kl']} diag kl {d['diag']['kl']}"); continue
        c = d["clean"]
        print(f"{n} drift {d.get('drift')}\n   clean ce {c['ce_clean_pl']}/{c['ce_clean_pl_sd']} q{c['ce_q05_50_95']} conf {c['conf']} "
              f"agree {c['agree']} ncls {c['ncls']} kl {c['kl']}\n   " +
              " | ".join(f"{k}: agr {d[k]['agree']} kl {d[k]['kl']} cecl {d[k].get('ce_clean_pl', '-')} conf {d[k]['conf']} ncls {d[k]['ncls']}"
                         for k in ("mild", "strong", "diag", "cm")))
    elif l.startswith(("==", "  steps", "teacher", "orig ", "qm[", "BJ DONE", "BC DONE", "BX DONE", "EXPORT", "Traceback")) or "Error" in l:
        print(l.rstrip()[:300])
