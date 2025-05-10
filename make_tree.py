#!/usr/bin/env python3
# visualize_prefix_trie.py

import sys
from graphviz import Digraph
from transformers import AutoTokenizer

def sanitize_label(text, max_len=40):
    """Escape quotes & newlines, then truncate."""
    t = text.replace('\n', '\\n').replace('"', '\\"')
    if len(t) > max_len:
        return t[:max_len] + "..."
    return t

def build_prefix_trie(tokens, tokenizer):
    """
    Recursively build a binary‐split trie. Each node is:
      [ start, end, decoded_substring, [child_nodes...] ]
    """
    def recurse(start, end):
        substr = tokenizer.decode(tokens[start:end], 
                                  skip_special_tokens=False)
        node = [start, end, substr, []]
        if end - start <= 1:
            return node
        mid = (start + end) // 2
        node[3].append(recurse(start, mid))
        node[3].append(recurse(mid, end))
        return node
    return recurse(0, len(tokens))

def visualize_trie(root, output_svg="prefix_trie.svg"):
    """
    Render the trie into an SVG via Graphviz, with sanitized labels.
    """
    g = Digraph("PrefixTrie", format="svg")
    g.attr("node", shape="box")
    counter = {"n": 0}

    def add(node, parent=None):
        idx = f"n{counter['n']}"
        counter["n"] += 1
        
        # sanitize & truncate the substring
        label = sanitize_label(node[2])
        g.node(idx, label)
        
        if parent is not None:
            g.edge(parent, idx)
        for child in node[3]:
            add(child, idx)

    add(root)
    g.render(output_svg.replace(".svg",""), cleanup=True)
    print(f"Wrote {output_svg}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_prefix_trie.py <text_or_path>")
        sys.exit(1)

    inp = sys.argv[1]
    # load literal text or file
    try:
        text = open(inp, encoding="utf-8").read()
    except FileNotFoundError:
        text = inp

    # build tokenizer & tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # build + visualize
    root = build_prefix_trie(tokens, tokenizer)
    visualize_trie(root)