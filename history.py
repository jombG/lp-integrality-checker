from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from models import Feedback, Instance, SolveResult, VerificationResult


def _serialize_feedback(feedback: Feedback) -> dict:
    inst = feedback.instance
    sr = feedback.solve_result
    vr = feedback.verification
    return {
        "iteration": feedback.iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instance": asdict(inst),
        "status": sr.status,
        "objective": sr.objective_value,
        "is_integer": vr.is_integer,
        "is_counterexample": vr.is_counterexample,
        "non_integer_vars": vr.non_integer_vars,
    }


def write_feedback(feedback: Feedback, path: str | Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_serialize_feedback(feedback), ensure_ascii=True) + "\n")


def load_history(path: str | Path) -> list[Feedback]:
    p = Path(path)
    if not p.exists():
        return []
    feedbacks: list[Feedback] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        inst_data = entry["instance"]
        instance = Instance(
            n=inst_data["n"],
            d=inst_data["d"],
            p=inst_data["p"],
            w=inst_data["w"],
            r=inst_data["r"],
        )
        solve_result = SolveResult(
            status=entry["status"],
            objective_value=entry.get("objective"),
        )
        verification = VerificationResult(
            is_integer=entry["is_integer"],
            is_counterexample=entry["is_counterexample"],
            non_integer_vars=entry.get("non_integer_vars", {}),
        )
        feedbacks.append(Feedback(
            iteration=entry["iteration"],
            instance=instance,
            solve_result=solve_result,
            verification=verification,
        ))
    return feedbacks
