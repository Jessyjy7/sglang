# dump_prompt_tree.py
from sglang import set_default_backend
from sglang.frontend.analysis import visualize_radix_tree, extract_radix_attention_tree
from sglang.runtime import RuntimeEndpoint
import sys

if len(sys.argv) != 2:
    print("Usage: python dump_prompt_tree.py  PATH_TO_PROMPT_FILE")
    sys.exit(1)

prompt_file = sys.argv[1]

# point at your running SGLang server
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# fetch & render the tree
visualize_radix_tree(
    prompt_file,
    output_path="prompt_attention_tree.svg"
)
print("Wrote prompt_attention_tree.svg for", prompt_file)