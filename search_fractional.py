from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

from pyomo.environ import value
from pyomo.opt import TerminationCondition

from model17 import Instance, build_model, solve_model


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def random_instance(n: int, p: int, w_min: int, w_max: int, r_max: int) -> Instance:
    d = n * p
    # Keep r_i in a range that is often feasible for constraints (14) with d = n*p.
    upper_r = min(r_max, d - p)
    w = [random.randint(w_min, w_max) for _ in range(n)]
    r = [random.randint(0, upper_r) for _ in range(n)]
    return Instance(n=n, d=d, p=p, w=w, r=r)


def grid_instances(
    n: int,
    p: int,
    w_values: Sequence[int],
    r_values: Sequence[int],
    max_instances: int,
) -> Iterable[Instance]:
    d = n * p
    valid_r = [v for v in r_values if 0 <= v <= d - 1]
    yielded = 0
    for w in itertools.product(w_values, repeat=n):
        for r in itertools.product(valid_r, repeat=n):
            yield Instance(n=n, d=d, p=p, w=list(w), r=list(r))
            yielded += 1
            if yielded >= max_instances:
                return


def fractionality_report(model, tol: float) -> Tuple[int, int, float, List[Tuple[str, int, int, float]]]:
    fractional = 0
    total = 0
    max_frac = 0.0
    sample: List[Tuple[str, int, int, float]] = []

    for i in model.I:
        for k in model.K:
            for name, var in (("x", model.x[i, k]), ("y", model.y[i, k])):
                v = float(value(var))
                frac = abs(v - round(v))
                max_frac = max(max_frac, frac)
                if frac > tol:
                    fractional += 1
                    if len(sample) < 10:
                        sample.append((name, int(i), int(k), v))
                total += 1
    return fractional, total, max_frac, sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a fractional optimum of LP-relaxation of task (17)."
    )
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--mode", choices=("random", "grid"), default="random")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-min", type=int, default=1)
    parser.add_argument("--w-max", type=int, default=20)
    parser.add_argument("--r-max", type=int, default=50)
    parser.add_argument("--w-values", default="1,2,3")
    parser.add_argument("--r-values", default="0,1,2,3,4,5")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tried = 0
    optimal = 0
    infeasible_or_other = 0

    if args.mode == "random":
        def stream():
            for _ in range(args.trials):
                yield random_instance(
                    n=args.n,
                    p=args.p,
                    w_min=args.w_min,
                    w_max=args.w_max,
                    r_max=args.r_max,
                )
        instances = stream()
    else:
        w_values = parse_int_list(args.w_values)
        r_values = parse_int_list(args.r_values)
        instances = grid_instances(
            n=args.n,
            p=args.p,
            w_values=w_values,
            r_values=r_values,
            max_instances=args.trials,
        )

    for inst in instances:
        tried += 1
        model = build_model(inst, integral=False)
        try:
            result = solve_model(model, solver_name=args.solver)
        except RuntimeError as exc:
            # appsi_highs may raise on infeasible instances before returning results.
            if "A feasible solution was not found" in str(exc):
                infeasible_or_other += 1
                if tried % args.progress_every == 0:
                    print(f"[{tried}] skipped, infeasible")
                continue
            raise

        if result.solver.termination_condition != TerminationCondition.optimal:
            infeasible_or_other += 1
            if tried % args.progress_every == 0:
                print(
                    f"[{tried}] skipped, termination={result.solver.termination_condition}"
                )
            continue

        optimal += 1
        fractional, total, max_frac, sample = fractionality_report(model, tol=args.tol)

        if fractional > 0:
            print("FOUND FRACTIONAL OPTIMUM")
            print("objective:", float(value(model.obj)))
            print("fractional_vars:", f"{fractional}/{total}")
            print("max_fractionality:", max_frac)
            print("sample_fractional_vars (name, i, k, value):")
            for row in sample:
                print(" ", row)
            print("instance:")
            print(" n =", inst.n, "p =", inst.p, "d =", inst.d)
            print(" w =", inst.w)
            print(" r =", inst.r)

            if args.out:
                payload = {
                    "instance": asdict(inst),
                    "objective": float(value(model.obj)),
                    "fractional_vars": fractional,
                    "total_vars": total,
                    "max_fractionality": max_frac,
                    "sample_fractional_vars": sample,
                }
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                print("saved:", args.out)
            return

        if tried % args.progress_every == 0:
            print(
                f"[{tried}] optimal LP but integral; objective={float(value(model.obj)):.6f}"
            )

    print("No fractional optimum found in search budget.")
    print("tried:", tried)
    print("optimal:", optimal)
    print("non-optimal/infeasible:", infeasible_or_other)


if __name__ == "__main__":
    main()
