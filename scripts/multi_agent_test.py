import sglang as sgl
from sglang.test.test_utils import select_sglang_backend, add_common_sglang_args_and_parse

# 1) single‐turn agent which you can parameterize by "role"
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that helps solve problems.")
    s += sgl.user(prompt)
    s += sgl.assistant(
        sgl.gen(
            "output",
            max_new_tokens=64,
            temperature=0.0,
        )
    )

def main(args):
    # 2) your shared prompt
    prompt_text = "Brainstorm three innovative applications of AI in education."

    # 3) the roles you want to compare/ensemble
    roles = [
        "visionary ideator",
        "practical engineer",
        "ethical reviewer",
    ]

    # 4) one call per role
    calls = [{"role": r, "prompt": prompt_text} for r in roles]

    # 5) wire up the backend & scheduler
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 6) run them *in parallel*, building one big radix tree
    results = agent_fn.run_batch(
        calls,
        schedule_policy="lpm",           # largest-prefix-most scheduler
        attention_backend="triton",      # radix kernels
        num_threads=len(roles),
        progress_bar=True,
    )

    # 7) sync & dump the merged attention/prefix tree
    state = results[0].state
    state.sync()
    state.dump_state_graph("multi_agent_tree.dot")
    print("Wrote multi_agent_tree.dot; render with:")
    print("    dot -Tpng multi_agent_tree.dot -o multi_agent_tree.png")

if __name__ == "__main__":
    parser = add_common_sglang_args_and_parse(argparse.ArgumentParser())
    main(parser.parse_args())
