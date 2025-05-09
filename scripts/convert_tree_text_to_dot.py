#!/usr/bin/env python3
import argparse
import re

def parse_blocks(lines):
    blocks, cur = [], []
    for l in lines:
        if l.startswith("==="):
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(l.rstrip("\n"))
    if cur:
        blocks.append(cur)
    return blocks

def make_dot(block, idx):
    dot = ["digraph G {", "  node [shape=box];"]
    stack, nid = [], 0
    for l in block:
        txt = re.sub(r"\x1B\[[0-9;]*[mK]", "", l)
        m = re.match(r"^( *)([├└│]+) (.*)$", txt)
        if not m: continue
        indent, label = len(m.group(1)), m.group(3).replace('"', r'\"')
        this = f"n{idx}_{nid}"; nid += 1
        dot.append(f'  {this} [label="{label}"];')
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            dot.append(f"  {stack[-1][1]} -> {this};")
        stack.append((indent, this))
    dot.append("}")
    return "\n".join(dot)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()

    with open(args.infile) as f:
        lines = f.readlines()
    blocks = parse_blocks(lines)
    if not blocks:
        print("ERROR: no tree blocks found"); return

    dot = make_dot(blocks[0], 0)
    with open(args.outfile, "w") as f:
        f.write(dot)
    print(f"Wrote {args.outfile} from block 0 of {len(blocks)}")

if __name__=="__main__":
    main()