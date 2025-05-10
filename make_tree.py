#!/usr/bin/env python3
from graphviz import Digraph
from transformers import AutoTokenizer
import sys

def build_prefix_trie(tokens, tokenizer):
    def recurse(start, end):
        text = tokenizer.decode(tokens[start:end], skip_special_tokens=False).replace("\\n", "\n")
        node = [start, end, text, []]
        if end - start <= 1:
            return node
        mid = (start + end) // 2
        node[3].append(recurse(start, mid))
        node[3].append(recurse(mid, end))
        return node
    return recurse(0, len(tokens))

def visualize_trie(root, output_svg="prefix_trie.svg"):
    g = Digraph("PrefixTrie", format="svg")
    counter = {"n": 0}

    def add(node, parent=None):
        idx = f"n{counter['n']}"
        counter["n"] += 1
        # label with the substring (newlines shown as actual breaks)
        g.node(idx, node[2] or "∅", shape="box")
        if parent:
            g.edge(parent, idx)
        for child in node[3]:
            add(child, idx)

    add(root)
    g.render(output_svg.replace(".svg", ""), cleanup=True)
    print(f"Wrote {output_svg}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_prefix_trie.py <text_or_path>")
        sys.exit(1)

    inp = sys.argv[1]
    try:
        text = open(inp, encoding="utf-8").read()
    except FileNotFoundError:
        text = inp

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    root = build_prefix_trie(tokens, tokenizer)
    visualize_trie(root)