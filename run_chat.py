#!/usr/bin/env python3
import json
from sglang import set_default_backend, function, system, user, assistant, gen
from tracing_runtime import TracingRuntime

# 1) point at your server with tracing enabled
backend = TracingRuntime("http://localhost:30000")
set_default_backend(backend)

@function
def chat(s, msg):
    s += system("You are a helpful assistant.")
    s += user(msg)
    s += assistant(gen("out", max_tokens=32))
    return s

# 2) two turns
state = chat("Hello! What can you do?")
state = chat("Okay, write me a story about a cat.")

# 3) save the two trees
for i, tree in enumerate(backend.trees, start=1):
    with open(f"tree{i}.json","w") as f:
        json.dump(tree, f, indent=2)
    print("Saved tree", i)