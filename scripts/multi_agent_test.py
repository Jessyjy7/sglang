#!/usr/bin/env python3
import argparse
import time

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend

# 1) Define a multi‐turn conversation per agent
@sgl.function
def agent_fn(s, role: str, prompt: str):
    s += sgl.system(f"You are the {role} agent collaborating in a multi‐agent system.")
    # Turn 1
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("first_reply", max_tokens=128, temperature=0.0))
    # Turn 2
    s += sgl.user("Please refine or expand on your first reply.")
    s += sgl.assistant(sgl.gen("second_reply", max_tokens=128, temperature=0.0))

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple agents in parallel and dump the full radix tree"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        required=True,
        help="List of agent roles, e.g. visionary_ideator practical_engineer ethical_reviewer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The shared prompt for all agents",
    )
    # adds --model-path, --device, --attention-backend, --schedule-policy, --port, etc.
    args = add_common_sglang_args_and_parse(parser)

    # 2) Wire up the backend (reads your --schedule-policy, --attention-backend, etc.)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 3) Build one call per role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 4) Run them all in parallel
    tic = time.time()
    states = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )
    elapsed = time.time() - tic
    print(f"✅ Completed in {elapsed:.2f}s")

    # 5) Sync and grab the shared radix‐attention tree from the first state
    state = states[0]
    state.sync()  # ensure all sub-processes have finished

    # 6) Extract the rustworkx.PyGraph and serialize to Graphviz DOT
    graph = state.graph        # <-- no .state here
    dot = graph.to_dot()
    with open("multi_agent_tree.dot", "w") as fout:
        fout.write(dot)
    print("→ Wrote multi_agent_tree.dot")

    print()
    print("Now render with Graphviz:")
    print("   dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()