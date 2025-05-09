#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.srt.utils import dump_state_graph

# 1) Single‐turn “agent” definition
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_tokens=64,   # correct param name
            temperature=0.0,
        )
    )

def main():
    # 2) CLI flags for your roles/prompt + common server args
    parser = argparse.ArgumentParser(
        description="Multi-agent radix-attention tree inspector"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["visionary_ideator", "practical_engineer", "ethical_reviewer"],
        help="List of agent roles to spin up",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Brainstorm three innovative applications of AI in education.",
        help="Shared prompt for all agents",
    )
    args = add_common_sglang_args_and_parse(parser)

    # 3) Wire up the backend (reads --model-path, --device, --schedule-policy, etc.)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 4) Build one call per agent role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 5) Fire them all in parallel (server already using LPM/Triton from its launch flags)
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )

    # 6) Grab the first ProgramState, and dump its radix‐attention tree
    state = results[0]
    dump_state_graph(state, "multi_agent_tree.dot")
    print("Wrote multi_agent_tree.dot; render locally with:")
    print("    dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()