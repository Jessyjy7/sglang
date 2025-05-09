#!/usr/bin/env python3
import argparse
import re

def parse_blocks(lines):
    """Split the dump into one block per agent turn."""
    blocks = []
    current = []
    for line in lines:
        if line.startswith("==="):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line.rstrip("\n"))
    if current:
        blocks.append(current)
    return blocks

def make_dot(block, idx):
    """
    Build a dot graph from one block of indent-tree text.
    Uses the box drawings ├─ └─ │ to infer parent→child.
    """
    dot = ["digraph G {", "  node [shape=box];"]
    stack = []  # (indent, node_id)
    node_id = 0

    for line in block:
        # strip ANSI color codes
        txt = re.sub(r"\x1B\[[0-9;]*[mK]", "", line)
        # match lines like "    ├─ Some text..."
        m = re.match(r"^( *)([├└│]+) (.*)$", txt)
        if not m:
            continue
        indent = len(m.group(1))
        label = m.group(3).replace('"', r'\"')

        # new node
        this_id = f"n{idx}_{node_id}"
        node_id += 1
        dot.append(f'  {this_id} [label="{label}"];')

        # find parent on stack
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent_id = stack[-1][1]
            dot.append(f"  {parent_id} -> {this_id};")

        stack.append((indent, this_id))

    dot.append("}")
    return "\n".join(dot)

def main():
    p = argparse.ArgumentParser(
        description="Convert SGLang dump_state_text output to Graphviz dot"
    )
    p.add_argument("--in", dest="infile", required=True,
                   help="Path to multi_agent_states.txt")
    p.add_argument("--out", dest="outfile", required=True,
                   help="Path to write multi_agent_tree.dot")
    args = p.parse_args()

    with open(args.infile) as f:
        lines = f.readlines()

    blocks = parse_blocks(lines)
    if not blocks:
        print("ERROR: no blocks found in input")
        return

    # Just convert the FIRST block (agent 0's two turns combined)
    dot = make_dot(blocks[0], idx=0)
    with open(args.outfile, "w") as f:
        f.write(dot)
    print(f"Wrote {args.outfile} (converted block0 of {len(blocks)})")

if __name__ == "__main__":
    main()