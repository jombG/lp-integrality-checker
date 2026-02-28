from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    RangeSet,
    SolverFactory,
    UnitInterval,
    Var,
    minimize,
    value,
)


@dataclass(frozen=True)
class Instance:
    n: int
    d: int
    p: int
    w: Sequence[int]
    r: Sequence[int]


def default_instance() -> Instance:
    w = [
        9,
        12,
        9,
        10,
        2,
        7,
        15,
        7,
        5,
        9,
        3,
        5,
        6,
        4,
        2,
        12,
        2,
        13,
        15,
        13,
        4,
        2,
        4,
        13,
        1,
        6,
        2,
        8,
        12,
        8,
    ]
    r = [
        0,
        0,
        1,
        11,
        9,
        0,
        11,
        8,
        11,
        1,
        6,
        5,
        2,
        14,
        10,
        0,
        5,
        13,
        9,
        7,
        8,
        2,
        5,
        12,
        14,
        6,
        11,
        6,
        12,
        5,
    ]
    n = len(w)
    p = 3
    d = n * p
    return Instance(n=n, d=d, p=p, w=w, r=r)


def build_model(inst: Instance, integral: bool = True) -> ConcreteModel:
    if len(inst.w) != inst.n or len(inst.r) != inst.n:
        raise ValueError("Lengths of w and r must be equal to n.")
    if inst.d != inst.n * inst.p:
        raise ValueError("This model assumes d = n * p.")

    m = ConcreteModel(name="Task17_QH")
    m.I = RangeSet(1, inst.n)
    m.K = RangeSet(1, inst.d)
    m.M = RangeSet(1, inst.p)
    m.S = RangeSet(0, inst.n - 1)

    # Boolean model (17) for integral=True, LP-relaxation for integral=False.
    domain = Binary if integral else UnitInterval
    m.x = Var(m.I, m.K, domain=domain)
    m.y = Var(m.I, m.K, domain=domain)

    # min sum_i sum_k w_i * y_ik
    m.obj = Objective(
        expr=sum(inst.w[i - 1] * m.y[i, k] for i in m.I for k in m.K),
        sense=minimize,
    )

    # (2) sum_i x_ik = 1, k = 1..d
    m.c2 = Constraint(m.K, rule=lambda mm, k: sum(mm.x[i, k] for i in mm.I) == 1)

    # (4) x_ik = 0, k <= r_i
    def c4_rule(mm: ConcreteModel, i: int, k: int):
        if k <= inst.r[i - 1]:
            return mm.x[i, k] == 0
        return Constraint.Skip

    m.c4 = Constraint(m.I, m.K, rule=c4_rule)

    # (14) sum_{s=0..n-1} x_{i, m + s*p} = 1, i in V, m = 1..p
    m.c14 = Constraint(
        m.I,
        m.M,
        rule=lambda mm, i, q: sum(mm.x[i, q + s * inst.p] for s in range(inst.n)) == 1,
    )

    # (15) sum_{l=s..n-1} x_{i, m + l*p} <= y_{i, m + s*p}
    m.c15 = Constraint(
        m.I,
        m.M,
        m.S,
        rule=lambda mm, i, q, s: sum(
            mm.x[i, q + l * inst.p] for l in range(s, inst.n)
        )
        <= mm.y[i, q + s * inst.p],
    )

    # (16) y_{i,k-1} >= y_{i,k}, k = 2..d
    def c16_rule(mm: ConcreteModel, i: int, k: int):
        if k >= 2:
            return mm.y[i, k - 1] >= mm.y[i, k]
        return Constraint.Skip

    m.c16 = Constraint(m.I, m.K, rule=c16_rule)

    return m


def solve_model(model: ConcreteModel, solver_name: str = "appsi_highs"):
    solver = SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' is not available. "
            "Install highspy and pyomo, or try solver_name='highs'."
        )
    return solver.solve(model, tee=False)


def extract_completion_times(model: ConcreteModel) -> List[int]:
    completion = []
    for i in model.I:
        c_i = 0
        for k in model.K:
            if value(model.y[i, k]) > 0.5:
                c_i = k
        completion.append(c_i)
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task (17) model from the Simanchev/Urazova paper using Pyomo + HiGHS."
    )
    parser.add_argument(
        "--solver",
        default="appsi_highs",
        help="Pyomo solver name (default: appsi_highs).",
    )
    parser.add_argument(
        "--show-x",
        action="store_true",
        help="Print all x[i,k] == 1 positions.",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Solve LP-relaxation (x,y in [0,1]) instead of binary model.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for checking integrality in --relax mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inst = default_instance()
    model = build_model(inst, integral=not args.relax)
    result = solve_model(model, solver_name=args.solver)

    print("status:", result.solver.status)
    print("termination:", result.solver.termination_condition)
    print("relaxation:", "LP" if args.relax else "MILP")
    print("objective:", float(value(model.obj)))

    completion = extract_completion_times(model)
    print("completion_times_Ci:", completion)

    if args.show_x:
        print("x[i,k] == 1:")
        for i in model.I:
            for k in model.K:
                if value(model.x[i, k]) > 0.5:
                    print(f"  i={i:2d}, k={k:2d}")

    if args.relax:
        max_fractionality = 0.0
        fractional_vars = 0
        total_vars = 0
        for i in model.I:
            for k in model.K:
                for var in (model.x[i, k], model.y[i, k]):
                    v = float(value(var))
                    frac = abs(v - round(v))
                    max_fractionality = max(max_fractionality, frac)
                    if frac > args.tol:
                        fractional_vars += 1
                    total_vars += 1
        print("fractional_vars:", f"{fractional_vars}/{total_vars}")
        print("max_fractionality:", max_fractionality)


if __name__ == "__main__":
    main()
