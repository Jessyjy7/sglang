# visualize_trie.py
from graphviz import Digraph

# 1) Define your exact tree as nested [start, end, children] lists:
tree = [0, 9, [
    [0, 4, [
        [0, 2, [
            [0, 1, []],
            [1, 2, []],
        ]],
        [2, 4, [
            [2, 3, []],
            [3, 4, []],
        ]],
    ]],
    [4, 9, [
        [4, 6, [
            [4, 5, []],
            [5, 6, []],
        ]],
        [6, 9, [
            [6, 7, []],
            [7, 9, [
                [7, 8, []],
                [8, 9, []],
            ]],
        ]],
    ]],
]]

# 2) Build a Graphviz Digraph
g = Digraph("RadixTrie", format="svg")
counter = {"n": 0}

def add_node(node, parent_id=None):
    idx = f"n{counter['n']}"
    counter["n"] += 1

    start, end, children = node
    # Label it exactly as in your printout
    g.node(idx, f"span=({start},{end})")

    if parent_id is not None:
        g.edge(parent_id, idx)

    for child in children:
        add_node(child, idx)

# 3) Populate the graph
add_node(tree)

# 4) Render to SVG (writes trie.svg)
g.render("trie", cleanup=True)
print("Wrote trie.svg")