from __future__ import annotations

from abc import ABC, abstractmethod

from models import Feedback, Instance


class OracleBase(ABC):
    @abstractmethod
    def generate_initial(self) -> Instance:
        ...

    @abstractmethod
    def generate_next(self, history: list[Feedback]) -> Instance:
        ...
