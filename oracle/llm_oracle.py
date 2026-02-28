from __future__ import annotations

import json
import logging
import random
import re
import time

from openai import OpenAI

from models import Feedback, Instance
from oracle.base import OracleBase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a mathematical optimization expert helping find counterexamples \
for an LP relaxation integrality conjecture.

## Problem
We study a Boolean scheduling model (Task 17, Simanchev & Urazova). The model has:
- n items to schedule across d = n * p time slots (p periods)
- Binary variables x[i,k] (assignment of item i to slot k) and y[i,k] (completion indicators)
- Objective: minimize sum_i sum_k w[i] * y[i,k], where w[i] are positive integer weights
- Release times r[i]: item i cannot be assigned to any slot k <= r[i]

Key constraints:
- (2)  Each slot gets exactly one item: sum_i x[i,k] = 1 for all k
- (4)  Release times: x[i,k] = 0 for k <= r[i]
- (14) Periodic assignment: sum_{s=0..n-1} x[i, m + s*p] = 1 for each item i and period m
- (15) Linking x to y: sum_{l=s..n-1} x[i, m+l*p] <= y[i, m+s*p]
- (16) Monotonicity of y: y[i,k-1] >= y[i,k]

## Goal
Find an instance where the LP relaxation (x,y in [0,1] instead of {0,1}) has a \
fractional optimal solution. Such instances are counterexamples to total dual \
integrality of the constraint system.

## Important context
The authors (Simanchev & Urazova, 2025) tested over 6000 instances with n from 10 to 200 \
and p in {3, 4, 10, 15, 20} using random parameters — all LP solutions were integer. \
A counterexample has NOT been found yet. You should try unusual structural patterns, \
not just random-looking parameters.

## Why fractionality might arise
Fractional LP solutions tend to appear when:
- Release times create tension: items compete for the same slots after their release
- Weights are heterogeneous: the optimizer "splits" items fractionally to reduce cost
- The problem is "tight": many items released at similar times with few available slots
- Try both small n (3-10) for exhaustive edge cases AND large n (20-100) for complex interactions
- Explore extreme or degenerate structures: all r[i] equal, r[i] very close to d-p, alternating patterns

## Instance format
You MUST respond with a single JSON object (no markdown, no explanation):
{
  "n": <int, number of items, >= 3>,
  "p": <int, number of periods, 2 to 5>,
  "w": [<positive ints>, length n],
  "r": [<non-negative ints>, length n, each r[i] <= n*p - p]
}
Note: d = n * p is computed automatically. Do NOT include d in your response.\
"""


class LLMOracle(OracleBase):
    def __init__(
        self,
        model: str = "o4-mini",
        temperature: float = 0.7,
        max_history_items: int = 10,
        max_retries: int = 2,
        request_delay: float = 5.0,
        n_range: tuple[int, int] = (3, 50),
        p_range: tuple[int, int] = (2, 5),
        w_min: int = 1,
        w_max: int = 20,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_history_items = max_history_items
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.n_range = n_range
        self.p_range = p_range
        self.w_min = w_min
        self.w_max = w_max

    def generate_initial(self) -> Instance:
        user_msg = (
            "Generate your first candidate instance. Start with a moderately sized "
            "instance (n=10..30, p=2..3) with heterogeneous weights and release times "
            "that create scheduling tension. Respond with only the JSON object."
        )
        return self._call_llm(user_msg)

    def generate_next(self, history: list[Feedback]) -> Instance:
        user_msg = self._format_history(history)
        return self._call_llm(user_msg)

    def _call_llm(self, user_msg: str) -> Instance:
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(self.request_delay)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "developer", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_completion_tokens=16384,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                text = content.strip()
                logger.info("LLM response: %s", text[:300])
                return self._parse_response(text)
            except Exception as exc:
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    continue

        logger.warning("All LLM attempts failed, falling back to random instance.")
        return self._random_fallback()

    def _parse_response(self, text: str) -> Instance:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        # Try direct parse first, then extract JSON object from surrounding text
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {cleaned[:200]}")
            data = json.loads(match.group())

        n = int(data["n"])
        p = int(data["p"])
        d = n * p
        w = [int(x) for x in data["w"]]
        r = [int(x) for x in data["r"]]

        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        if p < 1:
            raise ValueError(f"p must be positive, got {p}")
        if len(w) != n:
            raise ValueError(f"len(w)={len(w)} != n={n}")
        if len(r) != n:
            raise ValueError(f"len(r)={len(r)} != n={n}")
        if not all(x > 0 for x in w):
            raise ValueError(f"All weights must be positive: {w}")

        upper_r = max(0, d - p)
        r = [min(max(0, x), upper_r) for x in r]

        return Instance(n=n, d=d, p=p, w=w, r=r)

    def _format_history(self, history: list[Feedback]) -> str:
        recent = history[-self.max_history_items:]

        total = len(history)
        n_integer = sum(
            1 for f in history
            if f.verification.is_integer and f.solve_result.status == "optimal"
        )
        n_fractional = sum(1 for f in history if f.verification.is_counterexample)
        n_infeasible = sum(1 for f in history if f.solve_result.status != "optimal")

        lines = [
            f"## Search progress: {total} attempts so far",
            f"- Integer (no counterexample): {n_integer}",
            f"- Fractional (counterexample): {n_fractional}",
            f"- Infeasible/non-optimal: {n_infeasible}",
            "",
            "## Recent attempts (most recent last):",
        ]

        for fb in recent:
            inst = fb.instance
            sr = fb.solve_result
            vr = fb.verification

            if sr.status != "optimal":
                label = "INFEASIBLE"
            elif vr.is_counterexample:
                label = "FRACTIONAL"
            else:
                label = "INTEGER"

            entry = (
                f"- Iter {fb.iteration} [{label}]: "
                f"n={inst.n} p={inst.p} w={inst.w} r={inst.r}"
            )
            if sr.objective_value is not None:
                entry += f" obj={sr.objective_value:.4f}"
            if vr.non_integer_vars:
                entry += f" fractional_vars={len(vr.non_integer_vars)}"
            lines.append(entry)

        lines.append("")
        lines.append(
            "Based on the patterns above, propose a NEW instance that is likely to "
            "produce a fractional LP solution. Vary the parameters from previous attempts. "
            "Respond with only the JSON object."
        )
        return "\n".join(lines)

    def _random_fallback(self) -> Instance:
        n = random.randint(*self.n_range)
        p = random.randint(*self.p_range)
        d = n * p
        w = [random.randint(self.w_min, self.w_max) for _ in range(n)]
        upper_r = max(0, d - p)
        r = [random.randint(0, upper_r) for _ in range(n)]
        return Instance(n=n, d=d, p=p, w=w, r=r)
