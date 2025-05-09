#!/usr/bin/env python3
import argparse
import time

import sglang as sgl
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend
from sglang.utils import dump_state_text

# —————————————————————————————————————————————————————
# 1) Define your multi‐agent “function”
@sgl.function
def agent_fn(s, role: str, prompt: str):
    # each “agent” gets its own system prompt…
    s += sgl.system(f"You are the {role} agent, solve problems by delegation.")
    s += sgl.user(prompt)
    # …and each agent produces one assistant reply
    s += sgl.assistant(
        sgl.gen("output", max_new_tokens=64, temperature=0.0)
    )

# —————————————————————————————————————————————————————
# 2) Command‐line driver
def main():
    parser = argparse.ArgumentParser(
        description="Run N roles in parallel and dump their prefix tree"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        required=True,
        help="Space-separated list of agent roles, e.g. visionary engineer reviewer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The single prompt text to send to each role",
    )
    # this adds --device, --attention-backend, --schedule-policy, --port, etc.
    args = add_common_sglang_args_and_parse(parser)

    # pick up e.g. --schedule-policy lpm, --attention-backend triton, etc.
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # build the batch
    calls = [{"role": r, "prompt": args.prompt} for r in args.roles]

    # run them all in parallel
    tic = time.time()
    results = agent_fn.run_batch(
        calls,
        num_threads=len(calls),
        progress_bar=True,
    )
    latency = time.time() - tic

    print(f"✅ done in {latency:.2f}s, dumping state text…")
    # pull out the .state from each ProgramResult
    states = [r.state for r in results]

    # write a human‐readable dump of the entire radix tree + eviction log
    dump_state_text("multi_agent_states.txt", states)
    print("→ wrote   multi_agent_states.txt")

if __name__ == "__main__":
    main()