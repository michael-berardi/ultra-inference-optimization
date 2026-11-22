"""Engines that are named but not implemented yet. They refuse loudly instead of guessing."""
from __future__ import annotations

from ..errors import Refused
from .base import Engine


class _Stub(Engine):
    def _refuse(self, *a, **k):
        raise Refused("engine-not-implemented",
                      f"engine {self.name!r} is a documented stub in 0.1.0: tuning supports llama.cpp only "
                      f"(use `uio climb` for a generic metric)")

    info = baseline_args = levers = start = _refuse


class MlxLmEngine(_Stub):
    name = "mlx_lm.server"


class VllmEngine(_Stub):
    name = "vllm"
