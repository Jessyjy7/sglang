#!/usr/bin/env python3
import argparse

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
    dump_state_graph,         # <- pull in the Graphviz dumper
)

# 1) One‐turn “agent” function
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_tokens=64,
            temperature=0.0,
        )
    )

def main():
    # 2) CLI flags + common SGLang backend args
    parser = argparse.ArgumentParser(description="Multi‐agent radix tree inspector")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["visionary_ideator", "practical_engineer", "ethical_reviewer"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Brainstorm three innovative applications of AI in education.",
    )
    args = add_common_sglang_args_and_parse(parser)

    # 3) Hook up the server backend (reads --model-path, --device, --schedule-policy, etc.)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 4) Build and launch one “agent” per role in parallel
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )

    # 5) Dump the merged radix‐attention prefix tree to DOT
    state = results[0]     # ProgramState
    dump_state_graph(state, "multi_agent_tree.dot")
    print("Wrote multi_agent_tree.dot; render with:")
    print("   dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()