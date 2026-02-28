from __future__ import annotations

import random

from models import Feedback, Instance
from oracle.base import OracleBase


class RandomOracle(OracleBase):
    def __init__(
        self,
        n_range: tuple[int, int] = (200, 250),
        p_range: tuple[int, int] = (2, 5),
        w_min: int = 1,
        w_max: int = 20,
        seed: int | None = None,
    ):
        self.n_range = n_range
        self.p_range = p_range
        self.w_min = w_min
        self.w_max = w_max
        if seed is not None:
            random.seed(seed)

    def _generate(self) -> Instance:
        n = random.randint(*self.n_range)
        p = random.randint(*self.p_range)
        d = n * p
        w = [random.randint(self.w_min, self.w_max) for _ in range(n)]
        upper_r = max(0, d - p)
        r = [random.randint(0, upper_r) for _ in range(n)]
        return Instance(n=n, d=d, p=p, w=w, r=r)

    def generate_initial(self) -> Instance:
        return self._generate()

    def generate_next(self, history: list[Feedback]) -> Instance:
        return self._generate()
