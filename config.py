from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    max_iterations: int = 100
    integrality_tolerance: float = 1e-6
    history_file: str = "history.jsonl"
    solver_name: str = "appsi_highs"
    request_delay: float = 5.0
