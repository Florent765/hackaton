"""Export saved branch state dicts (interpolated from the original) with L.export.
usage: python B_export.py <saved_name>:<alpha>:<export_name> [...]"""
import sys
sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul"); sys.path.insert(0, "/p/home/jusers/jacquet2/jureca/ul/branches")
import ul_lib as L
import B_common as C

sd0 = L.original_state()
for spec in sys.argv[1:]:
    src, a, name = spec.split(":"); a = float(a)
    sd = L.interpolate(sd0, C.load_sd(src), a)
    L.export(sd, name, {"src": src, "alpha": a})
