#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend

# 1) Define your single-turn “agent” function
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_tokens=64,      # <–– use max_tokens, not max_new_tokens
            temperature=0.0,
        )
    )

def main():
    # 2) Parse roles & prompt (plus the common SGLang server flags)
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

    # 3) Hook up the HTTP/CUDA backend (reads --model-path, --device, --schedule-policy, etc.)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 4) Build one batch call per role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 5) Run them all in parallel (server already launched with LPM/Triton)
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )

    # 6) Sync & dump the merged radix-attention prefix tree
    result = results[0]
    result.sync()
    result.dump_state_graph("multi_agent_tree.dot")
    print("Wrote multi_agent_tree.dot; render locally with:")
    print("    dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()