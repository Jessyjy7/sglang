import sglang as sgl

# 1) define your per-agent function
@sgl.function
def agent_fn(s, role, prompt):
    s += sgl.system(f"You are a {role} agent that solves problems by delegation.")
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("output"))    # use defaults or supply your own sampling args

def main():
    roles  = ["visionary_ideator","practical_engineer","ethical_reviewer"]
    prompt = "Brainstorm three innovative applications of AI in education."
    calls  = [{"role":r,"prompt":prompt} for r in roles]

    # 2) run them in parallel so you get prefix sharing
    results = agent_fn.run_batch(
      calls,
      num_threads=len(roles),
      progress_bar=True,
      schedule_policy="lpm",           # radix scheduler
      attention_backend="triton",      # radix‐kernels-enabled backend
    )

    # 3) pick one result and sync
    res = results[0]
    res.sync()

    # 4) write the dot
    dot = res.state.graph.to_dot()
    with open("multi_agent_tree.dot","w") as f:
        f.write(dot)
    print("Wrote multi_agent_tree.dot")

if __name__=="__main__":
    main()