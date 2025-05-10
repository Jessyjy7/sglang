#!/usr/bin/env python3
import sys
import json
from graphviz import Digraph

def render(tree, name):
    dot = Digraph(name, format="svg")
    dot.attr("node", shape="box")
    cnt = 0

    def add(node, parent=None):
        nonlocal cnt
        nid = f"n{cnt}"; cnt += 1
        span = node["span"]
        dot.node(nid, f"{span[0]}→{span[1]}")
        if parent:
            dot.edge(parent, nid)
        for c in node.get("children", []):
            add(c, nid)

    add(tree)
    dot.render(name, cleanup=True)
    print(f"Wrote {name}.svg")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python render_tree.py path/to/tree.json")
        sys.exit(1)
    data = json.load(open(sys.argv[1]))
    tree = data["tree"]
    basename = sys.argv[1].rsplit(".",1)[0]
    render(tree, basename)