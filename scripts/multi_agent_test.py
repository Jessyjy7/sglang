#!/usr/bin/env python3
import argparse
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text

# —————————————————————————————————————————————————————
# 1) Define the per-agent, two-turn conversation
@sgl.function
def agent_fn(s, role: str, prompt: str):
    s += sgl.system(f"You are the {role} agent collaborating in a multi-agent system.")
    # Turn 1
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("first_reply", max_tokens=128, temperature=0.0))
    # Turn 2
    s += sgl.user("Please refine or expand on your first reply.")
    s += sgl.assistant(sgl.gen("second_reply", max_tokens=128, temperature=0.0))

# —————————————————————————————————————————————————————
# 2) Command-line driver
def main():
    parser = argparse.ArgumentParser(
        description="Run multiple agents in parallel and dump out the full radix tree."
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
    # this adds --model-path, --device, --attention-backend, --schedule-policy, --port, etc.
    args = add_common_sglang_args_and_parse(parser)

    # pick up your chosen backend flags (e.g. lpm + triton)
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # build the batch calls
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # run all agents in parallel
    tic = time.time()
    states = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )
    elapsed = time.time() - tic
    print(f"✅ Completed in {elapsed:.2f}s")

    # dump the chat *and* the full prefix/eviction tree into text
    dump_state_text("multi_agent_states.txt", states, show_tree=True)
    print("→ Wrote multi_agent_states.txt (with full tree)")

    print()
    print("Next, convert to DOT and render:")
    print("  python3 convert_tree_text_to_dot.py --in multi_agent_states.txt --out multi_agent_tree.dot")
    print("  dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    main()