"""Engine interface. An engine knows how to start a server for a profile + argument list, which
levers it offers, and how to stop the server it started."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Variant:
    label: str
    apply: Callable[[list[str]], list[str]]


@dataclass
class Lever:
    name: str
    variants: list[Variant]
    lossless: bool
    note: str = ""
    skip_reason: str | None = None  # set when the lever cannot apply here


class Server:
    """A running server. `pid` is None for engines that do not own a process (tests)."""
    base_url: str
    pid: int | None = None

    def alive(self) -> bool:  # pragma: no cover - interface
        return True

    def stop(self) -> None:  # pragma: no cover - interface
        pass

    def kill_for_watchdog(self) -> None:
        self.stop()


class Engine:
    name = "engine"

    def info(self) -> dict:
        raise NotImplementedError

    def baseline_args(self, profile: dict) -> list[str]:
        return list(profile.get("extraArgs") or [])

    def levers(self, baseline_args: list[str], ctx: int, machine: dict) -> list[Lever]:
        raise NotImplementedError

    def start(self, profile: dict, args: list[str], port: int, log_path) -> Server:
        raise NotImplementedError
