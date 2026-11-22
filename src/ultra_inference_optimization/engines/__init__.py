from ..errors import Refused
from .base import Engine, Lever, Server, Variant  # noqa: F401
from .llamacpp import LlamaCppEngine
from .stubs import MlxLmEngine, VllmEngine

ENGINES = {"llama.cpp": LlamaCppEngine, "mlx_lm.server": MlxLmEngine, "vllm": VllmEngine}


def get_engine(name: str, **kw) -> Engine:
    if name not in ENGINES:
        raise Refused("unknown-engine", f"unknown engine {name!r}; known: {', '.join(ENGINES)}")
    cls = ENGINES[name]
    return cls(**kw) if cls is LlamaCppEngine else cls()
