#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.srt.utils import dump_state_graph

# 1) Define a single-turn agent parameterized by role + prompt
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_new_tokens=64,
            temperature=0.0,
        )
    )

def main():
    # 2) Base parser for your roles & prompt
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

    # 3) Add the common SGLang backend args (model-path, device, port, etc.)
    args = add_common_sglang_args_and_parse(parser)

    # 4) Select & register the backend (must match your running server)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 5) Build one call per role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 6) Run all agents in parallel under the LPM (radix-tree) scheduler
    results = agent_fn.run_batch(
        calls,
        schedule_policy="lpm",        # radix: largest-prefix-most first
        attention_backend="triton",   # use Triton kernels
        num_threads=len(calls),
        progress_bar=True,
    )

    # 7) Dump the merged prefix/eviction tree to DOT
    dump_state_graph(results[0].state, "multi_agent_tree.dot")
    print("Wrote multi_agent_tree.dot; render with:")
    print("    dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()
