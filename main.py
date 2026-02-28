from __future__ import annotations

import argparse
import logging
import sys
import time

from config import Config
from history import load_history, write_feedback
from models import Feedback
from oracle.base import OracleBase
from resolver import solve
from verifier import verify


def run(oracle: OracleBase, config: Config) -> Feedback | None:
    history = load_history(config.history_file)
    start_iteration = len(history)

    if history:
        instance = oracle.generate_next(history)
    else:
        instance = oracle.generate_initial()

    try:
        for i in range(config.max_iterations):
            iteration = start_iteration + i
            print(f"[iter {iteration}] n={instance.n} d={instance.d} p={instance.p} "
                  f"w={instance.w} r={instance.r}")

            result = solve(instance, solver_name=config.solver_name)
            verification = verify(result, tolerance=config.integrality_tolerance)
            feedback = Feedback(
                iteration=iteration,
                instance=instance,
                solve_result=result,
                verification=verification,
            )
            history.append(feedback)
            write_feedback(feedback, config.history_file)

            print(f"  status={result.status}", end="")
            if result.objective_value is not None:
                print(f"  obj={result.objective_value:.4f}", end="")
            print(f"  integer={verification.is_integer}")

            if verification.is_counterexample:
                print("Counterexample found!")
                print(f"  Non-integer vars ({len(verification.non_integer_vars)}):")
                for var_name, val in list(verification.non_integer_vars.items())[:20]:
                    print(f"    {var_name} = {val}")
                return feedback

            if result.status != "optimal":
                print("  Skipping non-optimal instance.")

            time.sleep(config.request_delay)
            instance = oracle.generate_next(history)

        print(f"No counterexample found in {config.max_iterations} iterations.")
    except KeyboardInterrupt:
        completed = len(history) - start_iteration
        print(f"\nStopped by user after {completed} iterations.")
        print(f"History saved to {config.history_file} ({len(history)} total entries).")
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Counterexample finder for Task17 LP relaxation")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--history", default="history.jsonl")
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument(
        "--oracle",
        choices=["random", "llm"],
        default="random",
        help="Oracle implementation: 'random' or 'llm' (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="o4-mini",
        help="OpenAI model name for LLM oracle (default: o4-mini)",
    )
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between iterations (default: 5)")
    args = parser.parse_args()

    config = Config(
        max_iterations=args.max_iter,
        integrality_tolerance=args.tol,
        history_file=args.history,
        solver_name=args.solver,
        request_delay=args.delay,
    )

    if args.oracle == "llm":
        try:
            from oracle.llm_oracle import LLMOracle
            oracle: OracleBase = LLMOracle(model=args.model)
        except ImportError:
            print("LLM oracle requires 'openai' package. Install with: pip install openai")
            sys.exit(1)
    else:
        from oracle.random_oracle import RandomOracle
        oracle = RandomOracle()

    run(oracle, config)


if __name__ == "__main__":
    main()
