from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class EvolutionConfig:
    model: str
    reference_count: ClassVar[int] = 1
    max_rounds: ClassVar[int] = 8
    attack_max_attempts: ClassVar[int] = 3
    repair_max_attempts: ClassVar[int] = 3
    format_max_attempts: ClassVar[int] = 2
    no_update_patience: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
