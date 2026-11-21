"""A01 probe: print the StegaStamp ONNX decoder nodes from dec_conv_4 to the flatten (ops, attributes, weight shapes)."""
import sys
import onnx
sys.path.insert(0, "/p/project1/training2666/jacquet2/code/experiments/common")
import rlib as R  # noqa: E402
m = onnx.load(str(R.P.TASK_ROOT / "models/stegastamp/stega_stamp_logits.onnx")); g = m.graph
init = {i.name: i.dims for i in g.initializer}
on = False
for n in g.node:
    nm = n.output[0]
    if "gen_decoder/dec_conv_4" in nm or "dec_conv_4" in n.name:
        on = True
    if on:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        print(n.op_type, n.name, "IN", [(i, list(init[i]) if i in init else "") for i in n.input], "OUT", list(n.output), attrs)
    if on and n.op_type == "Flatten":
        break
print("---\nval_miou: 0.000000")
