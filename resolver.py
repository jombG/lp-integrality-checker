from __future__ import annotations

from pyomo.environ import value
from pyomo.opt import TerminationCondition

from model17 import Instance as M17Instance
from model17 import build_model, solve_model
from models import Instance, SolveResult


def _to_m17(inst: Instance) -> M17Instance:
    return M17Instance(n=inst.n, d=inst.d, p=inst.p, w=inst.w, r=inst.r)


def solve(instance: Instance, solver_name: str = "appsi_highs") -> SolveResult:
    m17_inst = _to_m17(instance)
    model = build_model(m17_inst, integral=False)

    try:
        result = solve_model(model, solver_name=solver_name)
    except RuntimeError as exc:
        if "A feasible solution was not found" in str(exc):
            return SolveResult(status="infeasible")
        raise

    tc = result.solver.termination_condition
    if tc != TerminationCondition.optimal:
        status = "infeasible" if tc == TerminationCondition.infeasible else "other"
        return SolveResult(status=status)

    solution_x: dict[tuple[int, int], float] = {}
    solution_y: dict[tuple[int, int], float] = {}
    for i in model.I:
        for k in model.K:
            solution_x[(i, k)] = float(value(model.x[i, k]))
            solution_y[(i, k)] = float(value(model.y[i, k]))

    return SolveResult(
        status="optimal",
        objective_value=float(value(model.obj)),
        solution_x=solution_x,
        solution_y=solution_y,
    )
