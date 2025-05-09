#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.utils import dump_state_text
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend

# 1) Define a one‐turn “agent” that just reads a role + prompt
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_tokens=64,      # correct arg name
            temperature=0.0,
        )
    )

def main():
    # 2) Parse roles + prompt, plus the usual SGLang server flags
    parser = argparse.ArgumentParser(description="Multi‐agent state dumper")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["visionary_ideator", "practical_engineer", "ethical_reviewer"],
        help="List of agent roles to spawn",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Brainstorm three innovative applications of AI in education.",
        help="Shared prompt for all agents",
    )
    args = add_common_sglang_args_and_parse(parser)

    # 3) Hook up the HTTP/CUDA backend you already started
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 4) Build one call per role
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # 5) Run them in parallel (server was launched with --schedule-policy lpm, --attention-backend triton)
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )

    # 6) Dump *all* agents' internal prefix/cache states to a text file
    dump_state_text("multi_agent_states.txt", results)
    print("✅ Wrote multi_agent_states.txt – inspect or post‐process for your tree diagram")

if __name__ == "__main__":
    main()