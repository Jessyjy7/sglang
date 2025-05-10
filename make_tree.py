# make_trie.py
import sys
from transformers import AutoTokenizer

def build_trie(tokens):
    # Each node is (start, end, [children])
    root = (0, len(tokens), [])
    def split(node):
        start, end, children = node
        length = end - start
        # stop splitting when the span is just 1 token
        if length <= 1:
            return
        # split in half to mimic a radix‐cache page
        mid = (start + end) // 2
        left = [start, mid, []]
        right = [mid, end, []]
        node[2].extend([left, right])
        split(left)
        split(right)
    split(root)
    return root

def print_trie(node, depth=0):
    start, end, children = node
    print("  " * depth + f"span=({start},{end})")
    for c in children:
        print_trie(c, depth + 1)

if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python make_trie.py <prompt_or_file>")
        sys.exit(1)
    inp = sys.argv[1]
    # if it’s an actual file on disk, read it; otherwise treat it literally
    try:
        text = open(inp, encoding="utf-8").read()
    except FileNotFoundError:
        text = inp

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    trie = build_trie(tokens)
    print_trie(trie)