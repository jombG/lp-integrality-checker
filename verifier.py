from __future__ import annotations

from models import SolveResult, VerificationResult


def verify(result: SolveResult, tolerance: float = 1e-6) -> VerificationResult:
    if result.status != "optimal" or result.solution_x is None:
        return VerificationResult(
            is_integer=True,
            is_counterexample=False,
        )

    non_integer_vars: dict[str, float] = {}

    for (i, k), val in result.solution_x.items():
        if abs(val - round(val)) > tolerance:
            non_integer_vars[f"x[{i},{k}]"] = val

    for (i, k), val in result.solution_y.items():
        if abs(val - round(val)) > tolerance:
            non_integer_vars[f"y[{i},{k}]"] = val

    is_integer = len(non_integer_vars) == 0
    return VerificationResult(
        is_integer=is_integer,
        is_counterexample=not is_integer,
        non_integer_vars=non_integer_vars,
    )
