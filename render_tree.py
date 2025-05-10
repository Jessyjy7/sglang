#!/usr/bin/env python3
# ~/sglang/render_tree.py

import sys, json
from graphviz import Digraph

def render(tree, name):
    g = Digraph(name, format="svg")
    g.attr("node", shape="box")
    cnt = 0
    def add(node, parent=None):
        nonlocal cnt
        nid = f"n{cnt}"; cnt += 1
        span = node["span"]
        g.node(nid, f"{span[0]}→{span[1]}")
        if parent: g.edge(parent, nid)
        for c in node.get("children", []):
            add(c, nid)
    add(tree)
    g.render(name, cleanup=True)
    print(f"Wrote {name}.svg")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: render_tree.py path/to/tree.json")
        sys.exit(1)
    data = json.load(open(sys.argv[1]))
    render(data["tree"], sys.argv[1].rsplit(".",1)[0])