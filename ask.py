# ask.py
from sglang import set_default_backend, function, system, user, assistant, gen
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

# point the frontend at your local SGLang server
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@function
def basic_qa(s, question):
    s += system("You are a helpful assistant.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=128))
    return s

if __name__ == "__main__":
    q = "What is the capital of France?"
    state = basic_qa(q)
    print("Q:", q)
    print("A:", state["answer"])