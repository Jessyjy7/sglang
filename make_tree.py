#!/usr/bin/env python3
# make_tree.py

import sys
from graphviz import Digraph
from transformers import AutoTokenizer

def sanitize_label(s: str, max_len=40) -> str:
    # 1) escape backslashes first
    s = s.replace("\\", "\\\\")
    # 2) escape quotes and newlines
    s = s.replace('"', '\\"').replace("\n", "\\n")
    # 3) truncate, but avoid ending on a single backslash
    if len(s) > max_len:
        s = s[:max_len]
        while s.endswith("\\"):
            s = s[:-1]
        s += "..."
    return s

def build_trie(tokens, tokenizer):
    """
    Build a binary-split trie. Node: [start, end, raw_substr, [children...]]
    """
    def rec(start, end):
        raw = tokenizer.decode(tokens[start:end], skip_special_tokens=False)
        node = [start, end, raw, []]
        if end - start <= 1:
            return node
        mid = (start + end) // 2
        node[3].append(rec(start, mid))
        node[3].append(rec(mid, end))
        return node
    return rec(0, len(tokens))

def visualize_trie(root, output_svg="prefix_trie.svg"):
    g = Digraph("RadixTrie", format="svg")
    g.attr("node", shape="box")
    counter = {"n": 0}

    def add(node, parent=None):
        idx = f"n{counter['n']}"
        counter["n"] += 1

        label = sanitize_label(node[2])
        g.node(idx, label)

        if parent is not None:
            g.edge(parent, idx)
        for child in node[3]:
            add(child, idx)

    add(root)
    g.render(output_svg.replace(".svg", ""), cleanup=True)
    print(f"Wrote {output_svg}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_tree.py <text-or-path>")
        sys.exit(1)

    inp = sys.argv[1]
    try:
        text = open(inp, encoding="utf-8").read()
    except FileNotFoundError:
        text = inp

    tk = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokens = tk.encode(text, add_special_tokens=False)

    root = build_trie(tokens, tk)
    visualize_trie(root)