# tracing_runtime.py
import json
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

class TracingRuntime(RuntimeEndpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We'll accumulate trees here
        self.trees = []

    def _extract_tree(self):
        # The RuntimeEndpoint stores its radix cache in self._client._radix_cache
        # or similar – we need to poke into its private fields.
        cache = getattr(self._client, "radix_cache", None)
        if cache is None:
            # Fallback lookup
            cache = getattr(self._client, "_radix_cache", None)
        if cache is None:
            raise RuntimeError("Cannot find radix_cache on client")

        # cache is a Trie object with .root and .serialize() or similar
        if hasattr(cache, "root"):
            tree = cache.root.to_dict()  # may differ by version
        elif hasattr(cache, "serialize"):
            tree = cache.serialize()
        else:
            # Last resort:‌ walk the internal nodes
            tree = cache.to_dict()
        self.trees.append(tree)
        return tree

    def chat(self, **kwargs):
        resp = super().chat_completions(**kwargs)
        self._extract_tree()
        return resp