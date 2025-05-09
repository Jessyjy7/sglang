#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.utils import dump_state_text
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend

# 1) Define your one‐turn “agent” function
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_tokens=64,    # correct parameter name
            temperature=0.0,
        )
    )

def main():
    # 2) Parse CLI args for roles & prompt, plus the usual SGLang server flags
    parser = argparse.ArgumentParser(description="Multi‐agent state dumper")
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

    # 3) Hook up the same backend your server is already running with
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 4) Build one call per role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 5) Fire them all in parallel (server was launched with --schedule-policy lpm, --attention-backend triton)
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )

    # 6) Dump every agent’s internal state (prefix cache + tokens) to a text log
    dump_state_text("multi_agent_states.txt", results)
    print("Wrote multi_agent_states.txt – inspect this to see every prefix node, eviction, and reuse.")

if __name__ == "__main__":
    main()