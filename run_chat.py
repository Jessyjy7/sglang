#!/usr/bin/env python3
import json
from sglang import set_default_backend, function, system, user, assistant, gen
from sglang.runtime import RuntimeEndpoint
from sglang.frontend.analysis import extract_radix_attention_tree

# 1) point at your live server
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# 2) define a simple chat function
@function
def chat(s, msg):
    s += system("You are a helpful assistant.")
    s += user(msg)
    s += assistant(gen("out", max_tokens=32))
    return s

# 3) Send “turns” and dump the tree after each
state = chat("Hello! What can you do?")
tree1 = extract_radix_attention_tree("")    # grabs the live cache‐tree
with open("tree1.json","w") as f:
    json.dump({"analysis":"radix_attention_tree","tree":tree1}, f, indent=2)
print("Wrote tree1.json")

state = chat("Okay, write me a story about a cat.")
tree2 = extract_radix_attention_tree("")
with open("tree2.json","w") as f:
    json.dump({"analysis":"radix_attention_tree","tree":tree2}, f, indent=2)
print("Wrote tree2.json")