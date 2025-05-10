#!/usr/bin/env python3
# ~/sglang/run_chat.py

import json
from sglang import set_default_backend, function, system, user, assistant, gen
# correct import path for RuntimeEndpoint
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

# if you have analysis support via pip (sglang[all]), you can do:
# from sglang.analysis import extract_radix_attention_tree
# otherwise skip those lines and use HTTP directly

# 1) point at your running server
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# 2) define a simple chat function
@function
def chat(s, msg):
    s += system("You are a helpful assistant.")
    s += user(msg)
    s += assistant(gen("out", max_tokens=32))
    return s

# 3) do two turns, dumping the radix tree after each
state = chat("Hello! What can you do?")
# if you have the helper, uncomment the next two lines:
# tree1 = extract_radix_attention_tree("")
# with open("tree1.json","w") as f: json.dump(tree1, f, indent=2)

state = chat("Okay, write me a story about a cat.")
# tree2 = extract_radix_attention_tree("")
# with open("tree2.json","w") as f: json.dump(tree2, f, indent=2)

# Fallback: if you don’t have extract_radix…, you can instead hit the HTTP API:
import requests
for i, fname in enumerate(("tree1.json","tree2.json"), start=1):
    resp = requests.post(
      "http://localhost:30000/analysis",
      json={"analysis":"radix_attention_tree","file":""}
    )
    resp.raise_for_status()
    open(fname,"w").write(resp.text)
    print("Wrote", fname)