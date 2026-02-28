from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instance:
    n: int
    d: int
    p: int
    w: list[int]
    r: list[int]


@dataclass(frozen=True)
class SolveResult:
    status: str  # "optimal" | "infeasible" | "other"
    objective_value: float | None = None
    solution_x: dict[tuple[int, int], float] | None = None
    solution_y: dict[tuple[int, int], float] | None = None


@dataclass(frozen=True)
class VerificationResult:
    is_integer: bool
    is_counterexample: bool
    non_integer_vars: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    iteration: int
    instance: Instance
    solve_result: SolveResult
    verification: VerificationResult
